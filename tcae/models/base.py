import torch
import torch.nn as nn
import torch.utils.checkpoint
import math
from typing import Optional, Tuple, Union, Callable
from torch import Tensor
import torch.nn.functional as F

from transformers.utils import is_flash_attn_2_available
if is_flash_attn_2_available():
    from flash_attn import flash_attn_func


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    @torch.compile
    def forward(self, x: Tensor) -> Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class PatchEmbed2D(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        num_channels: int = 3,
        hidden_size: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.hidden_size = hidden_size

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_channels = hidden_states.shape[1]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )

        embeddings = self.projection(hidden_states).flatten(2).transpose(1, 2)
        return embeddings


class VisionMLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        in_features = out_features = config.hidden_size
        hidden_features = int(config.hidden_size * config.mlp_ratio)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.activation = nn.functional.gelu
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.fc1(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.fc2(hidden_state)
        return hidden_state


class VisionAttention(nn.Module):
    def __init__(self, dim: int, num_attention_heads: int = 16) -> None:
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.head_dim = dim // num_attention_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        B, N, C = hidden_states.shape
        
        use_flash_attn = hidden_states.dtype == torch.bfloat16
        
        if use_flash_attn:
            # qkv shape: [B, N, 3, num_heads, head_dim]
            qkv = self.qkv(hidden_states).reshape(B, N, 3, self.num_attention_heads, self.head_dim)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            attn_output = flash_attn_func(q, k, v) 
            attn_output = attn_output.reshape(B, N, C)
        else:
            q, k, v = self.qkv(hidden_states).reshape(B, N, 3, self.num_attention_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
            
            attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)  # [B, heads, N, N]
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(1, 2).contiguous().reshape(B, N, C)
        
        attn_output = self.proj(attn_output)
        return attn_output


class VisionAttentionRope(nn.Module):
    def __init__(self, dim: int, num_attention_heads: int = 16) -> None:
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.head_dim = dim // num_attention_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        half_head_dim = dim // num_attention_heads // 2
        hw_seq_len = 16
        self.rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, C = hidden_states.shape
        q, k, v = self.qkv(hidden_states).reshape(B, N, 3, self.num_attention_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        q= self.rope(q)
        k = self.rope(k)
        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(B, N, C)
        attn_output = self.proj(attn_output)
        return attn_output



class LayerScale(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.lambda1 = nn.Parameter(config.layerscale_value * torch.ones(config.hidden_size))

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return hidden_state * self.lambda1

# Copied from transformers.models.beit.modeling_beit.drop_path
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class VisionLayer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        if not config.use_rmsnorm:
            self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.norm1 = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = VisionAttention(config.hidden_size, num_attention_heads=config.num_attention_heads)
        self.layer_scale1 = LayerScale(config)
        self.drop_path = DropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()
        if not config.use_rmsnorm:
            self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.norm2 = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        if not config.use_swiglu_ffn:
            self.mlp = VisionMLP(config)
        else:
            self.mlp = SwiGLUFFN(config.hidden_size, int(2/3 * config.hidden_size * config.mlp_ratio))
        self.layer_scale2 = LayerScale(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:

        if output_attentions:
            attn_output, attn_weights = self.attention(self.norm1(hidden_states), attention_mask, output_attentions=True)
        else:
            attn_output = self.attention(self.norm1(hidden_states), attention_mask, output_attentions=False)

        hidden_states = self.drop_path(self.layer_scale1(attn_output)) + hidden_states
        layer_output = self.drop_path(self.layer_scale2(self.mlp(self.norm2(hidden_states)))) + hidden_states

        if output_attentions:
            return layer_output, attn_weights
        else:
            return layer_output