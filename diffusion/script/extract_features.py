# Modified from:
#   RAE:    https://github.com/hustvl/LightningDiT/blob/main/extract_features.py


import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
import argparse
import os
import sys
from safetensors.torch import save_file
from datetime import datetime
from omegaconf import OmegaConf
from torchvision import transforms
import numpy as np
from PIL import Image

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, project_root)
from diffusion.img_latent_dataset import ImgLatentDataset
from tcae.models.config import VitConfig
from tcae.models.tcae import TCAE
from tcae.utils import remove_module_prefix



def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def main(args):
    """
    Run tokenizer on full dataset and cache the latent features.
    """
    assert torch.cuda.is_available(), "Caching features currently requires at least one GPU."

    # Setup DDP:
    try:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        world_size = dist.get_world_size()
        seed = args.seed + rank
        if rank == 0:
            print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")
    except:
        print("Failed to initialize DDP. Running in local mode.")
        rank = 0
        device = 0
        world_size = 1
        seed = args.seed
    
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup output folders:
    output_dir = os.path.join('results/cached_latents', f'imagenet_train_{args.image_size}')
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Caching latents to: {output_dir}")

    # Build tokenizer
    config = OmegaConf.load(args.tokenizer_cfg_path)
    tokenizer_config = config.model
    encoderconfig = VitConfig(**tokenizer_config.vitencoder_config)
    decoderconfig = VitConfig(**tokenizer_config.vitdecoder_config)
    
    tokenizer = TCAE(
        latent_dim=tokenizer_config.latent_dim,
        compress_rate_total=tokenizer_config.compress_rate_total,
        compress_rate_stage1=tokenizer_config.compress_rate_stage1,
        compress_type=tokenizer_config.compress_type,
        encoderconfig=encoderconfig,
        decoderconfig=decoderconfig,
        use_ssl=tokenizer_config.use_ssl,
        ssl_cfg=tokenizer_config.ssl_cfg
    ).to(device).eval()
    
    for param in tokenizer.parameters():
        param.requires_grad = False
    
    # Load pretrained weights
    tokenizer_state_dict = tokenizer.state_dict()
    checkpoint = torch.load(args.tokenizer_ckpt_path, map_location='cpu')
    
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    pretrained_weights = remove_module_prefix(state_dict)
    
    # Check for mismatches (only on rank 0 to avoid clutter)
    if rank == 0:
        for name, param in pretrained_weights.items():
            if name in tokenizer_state_dict:
                if param.shape != tokenizer_state_dict[name].shape:
                    print(f"Mismatch in parameter: {name}")
                    print(f"\tModel shape: {tokenizer_state_dict[name].shape}, Weight shape: {param.shape}")
            else:
                print(f"Parameter from weights not found in model: {name}")
        
        for name, param in tokenizer_state_dict.items():
            if name not in pretrained_weights:
                print(f"Parameter in model not found in weights: {name}")
                print(f"\tModel shape: {param.shape}")
        
        print(f"Loading pretrained model from {args.tokenizer_ckpt_path}")
    
    tokenizer.load_state_dict(pretrained_weights, strict=True)

    # Setup transforms - one without flip, one with flip
    transform_no_flip = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(p=0.0),  # No flip
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    
    transform_flip = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(p=1.0),  # Always flip
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # Setup datasets
    datasets = [
        ImageFolder(args.data_path, transform=transform_no_flip),
        ImageFolder(args.data_path, transform=transform_flip)
    ]
    
    if rank == 0:
        print(f'Loaded {len(datasets[0])} images from {args.data_path}')
    
    # Setup samplers
    samplers = [
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=args.seed
        ) for dataset in datasets
    ]
    
    # Setup dataloaders
    loaders = [
        DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        ) for dataset, sampler in zip(datasets, samplers)
    ]
    
    total_data_in_loop = len(loaders[0].dataset)
    if rank == 0:
        print(f"Total data in one loop: {total_data_in_loop}")
        print(f"Batches per GPU: {len(loaders[0])}")

    # Process and cache latents
    run_images = 0
    saved_files = 0
    latents = []
    latents_flip = []
    labels = []
    
    for batch_idx, batch_data in enumerate(zip(*loaders)):
        current_batch_size = batch_data[0][0].shape[0]
        run_images += current_batch_size
        
        if run_images % 1000 == 0 and rank == 0:
            print(f'{datetime.now()} [Rank {rank}] Processing {run_images} of {total_data_in_loop} images')
        
        with torch.no_grad():
            for loader_idx, data in enumerate(batch_data):
                x = data[0].to(device)
                y = data[1]
                z = tokenizer.encode(x).detach().cpu()
                
                if len(z.shape) == 3:  # (B, hw, C)
                    B, hw, C = z.shape
                    h = w = int(hw ** 0.5)
                    z = z.reshape(B, h, w, C).permute(0, 3, 1, 2).contiguous()
                
                if batch_idx == 0 and rank == 0 and loader_idx == 0:
                    print(f'Latent shape: {z.shape}, dtype: {z.dtype}')
                
                if loader_idx == 0:
                    latents.append(z)
                    labels.append(y)
                else:
                    latents_flip.append(z)
        
        if len(latents) >= args.save_every // args.batch_size:
            latents = torch.cat(latents, dim=0)
            latents_flip = torch.cat(latents_flip, dim=0)
            labels = torch.cat(labels, dim=0)
            
            save_dict = {
                'latents': latents,
                'latents_flip': latents_flip,
                'labels': labels
            }
            
            if rank == 0:
                for key in save_dict:
                    print(f'{key}: {save_dict[key].shape}')
            
            save_filename = os.path.join(
                output_dir, 
                f'latents_rank{rank:02d}_shard{saved_files:03d}.safetensors'
            )
            save_file(
                save_dict,
                save_filename,
                metadata={
                    'total_size': f'{latents.shape[0]}', 
                    'dtype': f'{latents.dtype}', 
                    'device': f'{latents.device}'
                }
            )
            
            if rank == 0:
                print(f'Saved {save_filename}')
            
            latents = []
            latents_flip = []
            labels = []
            saved_files += 1
    
    # Save remaining latents
    if len(latents) > 0:
        latents = torch.cat(latents, dim=0)
        latents_flip = torch.cat(latents_flip, dim=0)
        labels = torch.cat(labels, dim=0)
        
        save_dict = {
            'latents': latents,
            'latents_flip': latents_flip,
            'labels': labels
        }
        
        if rank == 0:
            for key in save_dict:
                print(f'{key}: {save_dict[key].shape}')
        
        save_filename = os.path.join(
            output_dir, 
            f'latents_rank{rank:02d}_shard{saved_files:03d}.safetensors'
        )
        save_file(
            save_dict,
            save_filename,
            metadata={
                'total_size': f'{latents.shape[0]}', 
                'dtype': f'{latents.dtype}', 
                'device': f'{latents.device}'
            }
        )
        
        if rank == 0:
            print(f'Saved {save_filename}')


    dist.barrier()
    if rank == 0:
        dataset = ImgLatentDataset(output_dir, latent_norm=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='', help="Path to ImageNet training data")
    parser.add_argument("--image_size", type=int, default=256, help="Image size for center crop")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size per GPU")
    parser.add_argument("--save_every", type=int, default=10000, help="Save latents every N images")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tokenizer_cfg_path", type=str, default='', help="Path to tokenizer config")
    parser.add_argument("--tokenizer_ckpt_path", type=str, default='', help="Path to tokenizer checkpoint")
    args = parser.parse_args()
    main(args)

