# Copyright (c) 2026 inclusionAI
# SPDX-License-Identifier: MIT


"""
Single-pass computation of:
    1) gFID (generated set vs reference mu/sigma)
    2) Inception Score (generated set itself)
"""

import argparse
import os
import shutil
from glob import glob

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

from torchmetrics.image.fid import FrechetInceptionDistance, _compute_fid
try:
    from torchmetrics.image.inception import InceptionScore
except ImportError:
    from torchmetrics.image import InceptionScore


# ============================================================================
# Dataset & DataLoader utilities
# ============================================================================
class CustomImageDataset(Dataset):
    """Simple dataset to load all images from a folder."""
    
    def __init__(self, folder_path, transform=None):
        self.image_paths = []
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp'):
            self.image_paths.extend(glob(os.path.join(folder_path, ext)))
        if not self.image_paths:
            raise ValueError(f"No images found in directory: '{folder_path}'")
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img
        except Exception as e:
            print(f"Warning: Failed to read/process {path}, skipped. Error: {e}")
            return None


def collate_fn(batch):
    """Filter out None samples from DataLoader batch."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return torch.tensor([])
    return torch.stack(batch, dim=0)


# ============================================================================
# Core: simultaneous gFID + IS computation
# ============================================================================
@torch.no_grad()
def calculate_gfid_and_is_with_reference(
    generated_dir: str,
    reference_npz: str,
    batch_size: int,
    num_workers: int,
    device: str
):
    """
    Args:
        generated_dir: Directory containing generated images
        reference_npz: Pre-computed reference {mu, sigma}.npz
    """
    device = torch.device(device)

    # Load reference statistics
    print(f"Loading reference mu/sigma from {reference_npz}...")
    try:
        with np.load(reference_npz) as f:
            mu_ref = torch.from_numpy(f["mu"]).to(device)
            sigma_ref = torch.from_numpy(f["sigma"]).to(device)
        print("Reference statistics loaded successfully.")
    except Exception as e:
        print(f"!! Failed to load reference .npz: {e}")
        return

    # Initialize torchmetrics
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)
    is_metric = InceptionScore(splits=10).to(device)

    # Setup DataLoader
    transform = T.ToTensor()  # Convert to [0,1] float32 tensor
    dataset = CustomImageDataset(generated_dir, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda")
    )

    # Single-pass update for both metrics
    for imgs in tqdm(loader, desc="Processing generated images"):
        if imgs.nelement() == 0:  # All samples filtered out
            continue
        imgs_uint8 = (imgs * 255).to(torch.uint8).to(device)  # [0-255] uint8

        fid_metric.update(imgs_uint8, real=False)
        is_metric.update(imgs_uint8)

    # Compute gFID
    mu_gen = fid_metric.fake_features_sum / fid_metric.fake_features_num_samples
    sigma_gen = (
        fid_metric.fake_features_cov_sum / fid_metric.fake_features_num_samples
    ) - torch.outer(mu_gen, mu_gen)

    fid_val = _compute_fid(
        mu_gen.double(), sigma_gen.double(),
        mu_ref.double(), sigma_ref.double()
    ).item()

    # Compute IS
    is_mean, is_std = is_metric.compute()
    is_mean, is_std = is_mean.item(), is_std.item()

    # Print results
    print("\n" + "=" * 60)
    print(f"gFID : {fid_val:.4f}")
    print(f"IS   : {is_mean:.4f} ± {is_std:.4f}")
    print("=" * 60)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute gFID and Inception Score for generated images.")
    parser.add_argument('--generated_dir', required=True, help='Directory containing generated images')
    parser.add_argument('--reference_npz', required=True, help='Pre-computed reference mu/sigma.npz file')
    parser.add_argument('--batch-size', type=int, default=64, help='DataLoader batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader num_workers')
    parser.add_argument('--delete-dir-after-calculation', action='store_true', help='Delete generated image directory after calculation')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")

    if not os.path.isdir(args.generated_dir):
        raise FileNotFoundError(f"Generated directory not found: {args.generated_dir}")
    if not os.path.isfile(args.reference_npz):
        raise FileNotFoundError(f"Reference .npz file not found: {args.reference_npz}")

    calculate_gfid_and_is_with_reference(
        generated_dir=args.generated_dir,
        reference_npz=args.reference_npz,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device
    )

    # Optional: delete directory after computation
    if args.delete_dir_after_calculation:
        print(f"\n--delete-dir-after-calculation specified, removing {args.generated_dir}")
        try:
            shutil.rmtree(args.generated_dir)
            print("Deletion successful.")
        except OSError as e:
            print(f"Deletion failed: {e.strerror}")
