# Copyright (c) 2026 inclusionAI
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
from .base import VisionLayer, nonlinearity



class ViTBaseDecoder(nn.Module):
    def __init__(self, config, compress_type) -> None:
        super().__init__()

        
        self.config = config
        self.patch_size = config.patch_size
        
        self.num_tokens_stage1 = (config.image_size // (config.patch_size * 2))  ** 2
        self.num_tokens_stage2 = (config.image_size // config.patch_size) ** 2 
        
        num_layers_per_stage = config.num_hidden_layers - 6
        self.layers_stage1 = nn.ModuleList([VisionLayer(config) for _ in range(num_layers_per_stage)])
        self.layers_stage2 = nn.ModuleList([VisionLayer(config) for _ in range(config.num_hidden_layers - num_layers_per_stage)])
        
        self.spatial_upsample_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.upsample_type = compress_type  # 'conv' or 'pixel_shuffle'
        
        if self.upsample_type == 'conv':
            self.upsample_conv = nn.Sequential(
                nn.ConvTranspose2d(config.hidden_size, config.hidden_size, kernel_size=2, stride=2),
                nn.GELU(),
                nn.Conv2d(config.hidden_size, config.hidden_size, kernel_size=3, stride=1, padding=1),
            )
        elif self.upsample_type == 'pixel_shuffle':
            self.upsample_mlp = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size * 2),
                nn.GELU(),
                nn.Linear(config.hidden_size * 2, config.hidden_size * 4),
            )
        else:
            raise ValueError(f"Unknown upsample_type: {self.upsample_type}")
        
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        if not config.conv_proj:
            self.out_proj = nn.Linear(config.hidden_size, config.patch_size * config.patch_size * config.num_channels)
        else:
            self.out_proj = nn.Conv2d(config.hidden_size, config.patch_size * config.patch_size * config.num_channels, kernel_size=3, stride=1, padding=1)

        self.position_embeddings_stage1 = nn.Parameter(torch.randn(1, self.num_tokens_stage1, config.hidden_size))
        self.position_embeddings_stage2 = nn.Parameter(torch.randn(1, self.num_tokens_stage2, config.hidden_size))
        
        torch.nn.init.normal_(self.position_embeddings_stage1, std=self.config.initializer_range)
        torch.nn.init.normal_(self.position_embeddings_stage2, std=self.config.initializer_range)
        
        self.gradient_checkpointing = False
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def spatial_upsample(self, hidden_states):
        """
        Upsample patch tokens from (H/16, W/16) to (H/8, W/8) using two optional methods
        Input: [B, N1, C] where N1 = (H/16) * (W/16)
        Output: [B, N2, C] where N2 = (H/8) * (W/8)
        """
        B, N, C = hidden_states.shape
        H = W = int(N ** 0.5)  # H/16, W/16
        
        if self.upsample_type == 'conv':
            hidden_states = hidden_states.reshape(B, H, W, C).permute(0, 3, 1, 2)
            hidden_states = self.upsample_conv(hidden_states)
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(B, -1, C)
            
        elif self.upsample_type == 'pixel_shuffle':
            hidden_states = self.upsample_mlp(hidden_states)  # [B, N, 4C]
            hidden_states = hidden_states.reshape(B, H, W, 4 * C).permute(0, 3, 1, 2)  # [B, 4C, H, W]
            B, C, H, W = hidden_states.shape
            out_channels = C // 4
            hidden_states = hidden_states.reshape(B, out_channels, 2, 2, H, W)
            hidden_states = hidden_states.permute(0, 1, 4, 2, 5, 3)
            hidden_states = hidden_states.reshape(B, out_channels, 2*H, 2*W)
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(B, -1, out_channels)  # [B, 4N, C]
        
        return hidden_states

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.config.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def forward(
        self,
        hidden_states: torch.Tensor,
        ids_restore: torch.Tensor = None,
        output_attentions: bool = False
    ) -> torch.Tensor:
        
        B, N_vis, C_d = hidden_states.shape

        # ========== Stage 1: Patch Size 16 ==========
        hidden_states = hidden_states + self.position_embeddings_stage1  # [B, N_stage1, C_d]

        all_attentions = [] if output_attentions else None
        
        for layer in self.layers_stage1:
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer.__call__, hidden_states, None, output_attentions
                )
            else:
                layer_outputs = layer(hidden_states, None, output_attentions=output_attentions)

            if output_attentions:
                hidden_states, attn = layer_outputs
                all_attentions.append(attn)
            else:
                hidden_states = layer_outputs

        # ========== Spatial Upsampling ==========
        hidden_states = self.spatial_upsample_norm(hidden_states)
        hidden_states = self.spatial_upsample(hidden_states)
        hidden_states = hidden_states + self.position_embeddings_stage2

        # ========== Stage 2: Patch Size 8 ==========
        for layer in self.layers_stage2:
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer.__call__, hidden_states, None, output_attentions
                )
            else:
                layer_outputs = layer(hidden_states, None, output_attentions=output_attentions)

            if output_attentions:
                hidden_states, attn = layer_outputs
                all_attentions.append(attn)
            else:
                hidden_states = layer_outputs

        hidden_states = self.layernorm(hidden_states)

        if self.config.nonlinear:
            hidden_states = nonlinearity(hidden_states)
            
        if not self.config.conv_proj:
            hidden_states = self.out_proj(hidden_states)
        else:
            B, HW, C = hidden_states.shape
            hidden_states = hidden_states.reshape(B, int(HW ** 0.5), int(HW ** 0.5), C).permute(0, 3, 1, 2)
            hidden_states = self.out_proj(hidden_states).reshape(B, -1, HW).permute(0, 2, 1)

        images_out = self.unpatchify(hidden_states)

        if output_attentions:
            return images_out, all_attentions
        else:
            return images_out


    def plaindecode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, C, h, w = hidden_states.shape
        hidden_states = hidden_states.reshape(batch_size, C, -1).permute(0, 2, 1)
        return self.forward(hidden_states)