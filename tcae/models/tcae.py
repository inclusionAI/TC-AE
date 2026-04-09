# Copyright (c) 2026 inclusionAI
# SPDX-License-Identifier: MIT

from typing import Any, Dict

import torch
import torch.nn as nn
from tcae.models.encoder import ViTBaseEncoder
from tcae.models.decoder import ViTBaseDecoder
from torch.cuda.amp import autocast

from functools import partial
from scipy.stats import norm
import numpy as np
import pickle as pkl
import math
from einops import rearrange
import torch.nn.functional as F



class TCAE(nn.Module):
    def __init__(
        self, 
        latent_dim=128,
        patch_size=8,            # Spatial compression ratio in ViT patch embdding layer
        compress_rate_total=32,  # Spatial compression ratio from image to latent
        compress_rate_stage1=2,  # Spatial compression ratio within ViT blocks
        compress_type='conv',
        use_ssl=None,
        ssl_cfg=None,
        encoderconfig = None, 
        decoderconfig = None,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        
        self.compress_rate_pathify = patch_size
        self.compress_rate_total = compress_rate_total
        self.compress_rate_stage1 = compress_rate_stage1
        self.compress_rate_bottleneck = int(compress_rate_total//compress_rate_stage1//patch_size)

        self.encoder = ViTBaseEncoder(config=encoderconfig, compress_type=compress_type, use_ssl=use_ssl, ssl_cfg=ssl_cfg)
        self.decoder = ViTBaseDecoder(config=decoderconfig, compress_type=compress_type)

        self.enc_dim = encoderconfig.hidden_size
        self.dec_dim = decoderconfig.hidden_size

        self.enc_merger = nn.Linear(self.compress_rate_stage1**2 * self.enc_dim, self.latent_dim)
        self.dec_unmerger = nn.Linear(self.latent_dim, self.compress_rate_stage1**2 * self.dec_dim)

        params_En = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad) / 1e6
        params_De = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad) / 1e6
        print(f"Encoder: {params_En}M, Decoder: {params_De}M")


    def forward(self, recon_input, dino_input=None, dino_tea_temp=None, masks=None, output_attentions=False, labels=None):

        recon_input = recon_input.cuda(non_blocking=True)
        moments, output2 = self.encoder(recon_input)
        B, HW, C = moments.shape

        sub_pz = self.compress_rate_bottleneck
        H = W = int(math.sqrt(HW))
        x = moments.view(B, H, W, C)
        x = rearrange(x, 'b (h p1) (w p2) c -> b h w (p1 p2 c)', p1=sub_pz, p2=sub_pz)
        latent_before_compress = x.view(B, -1, self.compress_rate_bottleneck**2 * self.enc_dim)
        latent = self.enc_merger(latent_before_compress)

        x = self.dec_unmerger(latent)
        x = x.view(B, H // sub_pz, W // sub_pz, self.compress_rate_bottleneck**2 * self.dec_dim) 
        x = rearrange(x, 'b h w (p1 p2 c) -> b (h p1) (w p2) c', p1=sub_pz, p2=sub_pz, c=self.dec_dim)
        z = x.view(B, H * W, self.dec_dim)

        reconstruction = self.decoder(z)
        return reconstruction



    def encode(self, inp):
        moments, _, = self.encoder(inp)
        B, HW, C = moments.shape
        sub_pz = self.compress_rate_bottleneck
        H = W = int(math.sqrt(HW))
        x = moments.view(B, H, W, C)
        x = rearrange(x, 'b (h p1) (w p2) c -> b h w (p1 p2 c)', p1=sub_pz, p2=sub_pz)
        latent_before_compress = x.view(B, -1, self.compress_rate_bottleneck**2 * self.enc_dim)
        latent = self.enc_merger(latent_before_compress)
        return latent

    

    def load_state_dict(self, state_dict: Dict[str, Any], strict=True, assign=False):
        result = super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)

        if hasattr(result, 'missing_keys') and result.missing_keys:
            print('Missing keys:')
            for k in result.missing_keys:
                print('  ', k)
        if hasattr(result, 'unexpected_keys') and result.unexpected_keys:
            print('Unexpected keys:')
            for k in result.unexpected_keys:
                print('  ', k)
        if hasattr(result, 'error_msgs') and result.error_msgs:
            print('Error messages:')
            for msg in result.error_msgs:
                print('  ', msg)
        return result
        

