# Copyright (c) 2026 inclusionAI
# SPDX-License-Identifier: MIT

import os
import sys
import argparse
import torch
from PIL import Image
from omegaconf import OmegaConf
from torchvision.transforms import InterpolationMode, transforms

# Setup project path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)

from tcae.models.config import VitConfig
from tcae.models.tcae import TCAE
from tcae.utils import remove_module_prefix, get_image_paths
from tcae.dataset import pil_loader


def main(args):
    """Main function for image reconstruction."""
    # Setup
    device = f'cuda:{args.rank}'
    image_paths = get_image_paths(args.img_folder)
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Load model
    config = OmegaConf.load(args.config)
    model_config = config.model
    encoderconfig = VitConfig(**model_config.vitencoder_config)
    decoderconfig = VitConfig(**model_config.vitdecoder_config)
    
    model = TCAE(
        latent_dim=model_config.latent_dim,
        compress_rate_total=model_config.compress_rate_total,
        compress_rate_stage1=model_config.compress_rate_stage1,
        compress_type=model_config.compress_type,
        encoderconfig=encoderconfig,
        decoderconfig=decoderconfig,
        use_ssl=model_config.use_ssl,
        ssl_cfg=model_config.ssl_cfg
    ).to(device).eval()
    
    for param in model.parameters():
        param.requires_grad = False
    
    checkpoint = torch.load(args.ckpt_path, map_location='cpu')
    state_dict = checkpoint.get('model', checkpoint)
    new_state_dict = remove_module_prefix(state_dict)
    model.load_state_dict(new_state_dict, strict=True)
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size), interpolation=InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    
    # Process images
    for input_img_path in image_paths:
        image_name = os.path.splitext(os.path.basename(input_img_path))[0]
        
        # Load and transform image
        input_img_pil = pil_loader(input_img_path)
        input_img = transform(input_img_pil).unsqueeze(0).to(device)
        
        # Reconstruct
        with torch.no_grad():
            output_img = model(input_img)[0]
        
        # Denormalize and convert to PIL
        output_img = ((output_img + 1) / 2).clamp(0, 1).squeeze(0)
        output_img_pil = transforms.ToPILImage()(output_img)
        
        input_img = ((input_img[0] + 1) / 2).clamp(0, 1)
        input_img_pil = transforms.ToPILImage()(input_img)
        
        # Concatenate and save
        total_width = input_img_pil.width + output_img_pil.width
        max_height = max(input_img_pil.height, output_img_pil.height)
        concatenated_img = Image.new('RGB', (total_width, max_height))
        concatenated_img.paste(input_img_pil, (0, 0))
        concatenated_img.paste(output_img_pil, (input_img_pil.width, 0))
        
        output_path = os.path.join(args.output_folder, f'recon_{image_name}.png')
        concatenated_img.save(output_path)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reconstruct images using TCAE model')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--img_folder', type=str, required=True, help='Input image folder path')
    parser.add_argument('--output_folder', type=str, required=True, help='Output folder for reconstructed images')
    parser.add_argument('--rank', type=int, required=True, help='CUDA device rank')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--mid_reso', type=float, default=1.125, help='Middle resolution scaling factor')
    parser.add_argument('--img_size', type=int, default=256, help='Input image size')
    args = parser.parse_args()
    
    main(args)
