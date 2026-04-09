# Modified from:
#   RAE:    https://github.com/bytetriper/RAE/blob/main/src/stage2/models/DDT.py


from __future__ import annotations
from math import sqrt
from re import L
from regex import B
import torch
import torch.nn as nn

from transformers import SwinModel
import torch
from torch import nn
from .lightningDiT import PatchEmbed, Mlp, NormAttention
from timm.models.vision_transformer import PatchEmbed, Mlp
from .model_utils import VisionRotaryEmbeddingFast, RMSNorm, SwiGLUFFN, GaussianFourierEmbedding, LabelEmbedder, NormAttention, get_2d_sincos_pos_embed
import torch.nn.functional as F
from typing import *


def DiTModulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Applies per-segment modulation to x.

    Args:
        x: Tensor of shape (B, L_x, D)
        shift: Tensor of shape (B, L, D)
        scale: Tensor of shape (B, L, D)
    Returns:
        Tensor of shape (B, L_x, D): x * (1 + scale) + shift, 
        with shift and scale repeated to match L_x if necessary.
    """
    B, Lx, D = x.shape
    _, L, _ = shift.shape
    if Lx % L != 0:
        raise ValueError(f"L_x ({Lx}) must be divisible by L ({L})")
    repeat = Lx // L
    if repeat != 1:
        # repeat each of the L segments 'repeat' times along the length dim
        shift = shift.repeat_interleave(repeat, dim=1)
        scale = scale.repeat_interleave(repeat, dim=1)
    # apply modulation
    return x * (1 + scale) + shift


def DiTGate(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """
    Applies per-segment modulation to x.

    Args:
        x: Tensor of shape (B, L_x, D)
        gate: Tensor of shape (B, L, D)
    Returns:
        Tensor of shape (B, L_x, D): x * gate, 
        with gate repeated to match L_x if necessary.
    """
    B, Lx, D = x.shape
    _, L, _ = gate.shape
    if Lx % L != 0:
        raise ValueError(f"L_x ({Lx}) must be divisible by L ({L})")
    repeat = Lx // L
    if repeat != 1:
        # repeat each of the L segments 'repeat' times along the length dim
        # print(f"gate shape: {gate.shape}, x shape: {x.shape}")
        gate = gate.repeat_interleave(repeat, dim=1)
    # apply modulation
    return x * gate


class DiTBlock(nn.Module):
    """
    Lightning DiT Block. We add features including: 
    - ROPE
    - QKNorm 
    - RMSNorm
    - SwiGLU
    - No shift AdaLN.
    Not all of them are used in the final model, please refer to the paper for more details.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_qknorm=False,
        use_swiglu=True,
        use_rmsnorm=True,
        wo_shift=False,
        **block_kwargs
    ):
        super().__init__()

        # Initialize normalization layers
        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)

        # Initialize attention layer
        self.attn = NormAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            **block_kwargs
        )

        # Initialize MLP layer
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        def approx_gelu(): return nn.GELU(approximate="tanh")
        if use_swiglu:
            # here we did not use SwiGLU from xformers because it is not compatible with torch.compile for now.
            self.mlp = SwiGLUFFN(hidden_size, int(2/3 * mlp_hidden_dim))
        else:
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0
            )

        # Initialize AdaLN modulation
        if wo_shift:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 4 * hidden_size, bias=True)
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )
        self.wo_shift = wo_shift

    def forward(self, x, c, feat_rope=None):
        if len(c.shape) < len(x.shape):
            c = c.unsqueeze(1)  # (B, 1, C)
        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(
                c).chunk(4, dim=-1)
            shift_msa = None
            shift_mlp = None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
                c).chunk(6, dim=-1)
        x = x + DiTGate(self.attn(DiTModulate(self.norm1(x),
                        shift_msa, scale_msa), rope=feat_rope), gate_msa)
        x = x + DiTGate(self.mlp(DiTModulate(self.norm2(x),
                        shift_mlp, scale_mlp)), gate_mlp)
        return x


class DiTFinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels, use_rmsnorm=False):
        super().__init__()
        if not use_rmsnorm:
            self.norm_final = nn.LayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        if len(c.shape) < len(x.shape):
            c = c.unsqueeze(1)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = DiTModulate(self.norm_final(x), shift, scale)  # no gate
        x = self.linear(x)
        return x


class DiT(nn.Module):
    def __init__(
            self,
            input_size: int = 32,
            patch_size: Union[list, int] = 2,
            in_channels: int = 4,
            hidden_size: Union[list, int] = 1152,
            depth: Union[list, int] = 28,
            num_heads: Union[list[int], int] = 16,
            mlp_ratio=4.0,
            class_dropout_prob=0.1,
            num_classes=1000,
            use_qknorm=False,
            use_swiglu=True,
            use_rope=True,
            use_rmsnorm=True,
            wo_shift=False,
            use_pos_embed: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels

        # 兼容列表格式的config，只取第一个值（encoder部分）
        # 处理 OmegaConf ListConfig
        if isinstance(hidden_size, (list, tuple)) or hasattr(hidden_size, '__getitem__'):
            self.hidden_size = int(hidden_size[0])
        else:
            self.hidden_size = int(hidden_size)
            
        if isinstance(depth, (list, tuple)) or hasattr(depth, '__getitem__'):
            self.num_blocks = int(depth[0])
        else:
            self.num_blocks = int(depth)
            
        if isinstance(num_heads, (list, tuple)) or hasattr(num_heads, '__getitem__'):
            self.num_heads = int(num_heads[0])
        else:
            self.num_heads = int(num_heads)
            
        # 兼容列表格式的patch_size，只取第一个
        if isinstance(patch_size, (list, tuple)) or hasattr(patch_size, '__getitem__'):
            self.patch_size = int(patch_size[0])
        else:
            self.patch_size = int(patch_size)
            
        self.use_rope = use_rope
        
        # Calculate channels per token - 确保都是 int 类型
        x_channel_per_token = int(in_channels) * self.patch_size * self.patch_size
        self.x_channel_per_token = x_channel_per_token
        
        # Patch embedding
        self.x_embedder = PatchEmbed(
            int(input_size), 
            self.patch_size, 
            x_channel_per_token, 
            self.hidden_size, 
            bias=True
        )
        
        # Time and label embeddings
        self.t_embedder = GaussianFourierEmbedding(self.hidden_size)
        self.y_embedder = LabelEmbedder(
            int(num_classes), self.hidden_size, float(class_dropout_prob))
        
        # Final layer
        self.final_layer = DiTFinalLayer(
            self.hidden_size, 1, x_channel_per_token, use_rmsnorm=use_rmsnorm)
        
        # Position embedding
        self.use_pos_embed = use_pos_embed
        if use_pos_embed:
            num_patches = self.x_embedder.num_patches
            self.pos_embed = nn.Parameter(torch.zeros(
                1, num_patches, self.hidden_size), requires_grad=False)
        
        # Rotary position encoding
        if self.use_rope:
            half_head_dim = self.hidden_size // self.num_heads // 2
            hw_seq_len = int(sqrt(self.x_embedder.num_patches))
            self.feat_rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=hw_seq_len,
            )
        else:
            self.feat_rope = None
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                self.hidden_size,
                self.num_heads,
                mlp_ratio=float(mlp_ratio),
                use_qknorm=use_qknorm,
                use_rmsnorm=use_rmsnorm,
                use_swiglu=use_swiglu,
                wo_shift=wo_shift,
            ) for i in range(self.num_blocks)
        ])
        
        self.initialize_weights()


    def initialize_weights(self, xavier_uniform_init: bool = False):
        if xavier_uniform_init:
            def _basic_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            self.apply(_basic_init)
        
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        
        if self.use_pos_embed:
            # Initialize (and freeze) pos_embed by sin-cos embedding:
            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
            self.pos_embed.data.copy_(
                torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.x_channel_per_token
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x)
        
        if self.use_pos_embed:
            x = x + self.pos_embed
        
        t = self.t_embedder(t)
        y = self.y_embedder(y, self.training)
        c = nn.functional.silu(t + y)
        
        for block in self.blocks:
            x = block(x, c, feat_rope=self.feat_rope)
        
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale, cfg_interval=(0, 1)):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        
        guid_t_min, guid_t_max = cfg_interval
        assert guid_t_min < guid_t_max, "cfg_interval should be (min, max) with min < max"
        
        t = t[: len(t) // 2]  # get t for the conditional half
        half_eps = torch.where(
            ((t >= guid_t_min) & (t <= guid_t_max)).view(-1, *[1] * (len(cond_eps.shape) - 1)),
            uncond_eps + cfg_scale * (cond_eps - uncond_eps), 
            cond_eps
        )
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    def forward_with_autoguidance(self, x, t, y, cfg_scale, additional_model_forward, cfg_interval=(0, 1)):
        """
        Forward pass of DiT, but also contain the forward pass for the additional model
        """
        model_out = self.forward(x, t, y)
        ag_model_out = additional_model_forward(x, t, y)
        eps = model_out[:, :self.in_channels]
        ag_eps = ag_model_out[:, :self.in_channels]

        guid_t_min, guid_t_max = cfg_interval
        assert guid_t_min < guid_t_max, "cfg_interval should be (min, max) with min < max"
        
        eps = torch.where(
            ((t >= guid_t_min) & (t <= guid_t_max)).view(-1, *[1] * (len(eps.shape) - 1)),
            ag_eps + cfg_scale * (eps - ag_eps), 
            eps
        )

        return eps
