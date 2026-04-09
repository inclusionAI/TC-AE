# Copyright (c) 2026 inclusionAI
# SPDX-License-Identifier: MIT

import os
import sys
import argparse
import torch
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)

from tcae.models.config import VitConfig
from tcae.models.tcae import TCAE
from tcae.dataset import build_eval_datataset
from tcae.utils import remove_module_prefix


def main(args):
    device = f'cuda:{args.rank}'

    # Initialize model
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
    state_dict = remove_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=True)


    # Prepare evaluation dataset and dataloader
    eval_dataset = build_eval_datataset(
        args.dataset_root,
        args.img_size,
        mid_reso=args.mid_reso
    )

    dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        persistent_workers=True
    )


    # Initialize image quality metrics
    psnr_metric = PeakSignalNoiseRatio().to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)

    psnr_scores, ssim_scores, lpips_scores = [], [], []


    # Evaluation loop
    for batch in tqdm(dataloader, desc='Evaluating'):
        original_images, _ = batch
        original_images = original_images.to(device)
        
        with torch.no_grad():
            reconstructed_images = model(original_images)
            reconstructed_images = ((reconstructed_images + 1) / 2).clamp(0, 1)
            original_images = ((original_images + 1) / 2).clamp(0, 1)
            
            real_images = (original_images * 255).to(torch.uint8)
            fake_images = (reconstructed_images * 255).to(torch.uint8)
        
        psnr_scores.append(psnr_metric(reconstructed_images, original_images).item())
        ssim_scores.append(ssim_metric(reconstructed_images, original_images).item())
        lpips_scores.append(lpips_metric(reconstructed_images, original_images).item())
        fid_metric.update(real_images, real=True)
        fid_metric.update(fake_images, real=False)


    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)
    avg_lpips = np.mean(lpips_scores)
    fid_score = fid_metric.compute().item()


    # Print evaluation results
    print(f"Average PSNR: {avg_psnr:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average LPIPS: {avg_lpips:.4f}")
    print(f"FID Score: {fid_score:.4f}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate TCAE model reconstruction quality')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset_root', type=str, required=True, help='Root directory of evaluation dataset')
    parser.add_argument('--rank', type=int, required=True, help='CUDA device rank')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--mid_reso', type=float, default=1.125, help='Middle resolution scaling factor')
    parser.add_argument('--img_size', type=int, default=256, help='Input image size')
    parser.add_argument('--batch_size', type=int, default=250, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=12, help='Number of data loading workers')
    args = parser.parse_args()
    main(args)
