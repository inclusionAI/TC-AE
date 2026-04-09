# Modified from:
#   RAE: https://github.com/bytetriper/RAE/blob/main/src/utils/wandb_utils.py


import torch
from torchvision.utils import make_grid
import torch.distributed as dist
from PIL import Image
import os
import argparse
import hashlib
import math


def is_main_process():
    return dist.get_rank() == 0

def namespace_to_dict(namespace):
    return {
        k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
        for k, v in vars(namespace).items()
    }


def generate_run_id(exp_name):
    # https://stackoverflow.com/questions/16008670/how-to-hash-a-string-into-8-digits
    return str(int(hashlib.sha256(exp_name.encode('utf-8')).hexdigest(), 16) % 10 ** 8)


def log_image_local(sample, step=None, save_dir="./logs/images"):
    """Save batch of samples as local image file.
    Args:
        sample (torch.Tensor): Image batch [B, C, H, W], range 0~1
        step (int): Training step for filename
        save_dir (str): Save directory
    """
    if is_main_process():  # Only save on main process
        os.makedirs(save_dir, exist_ok=True)
        img_np = array2grid(sample)
        img_pil = Image.fromarray(img_np)

        if step is None:
            filename = "samples.png"
        else:
            filename = f"samples_step_{step}.png"

        save_path = os.path.join(save_dir, filename)
        img_pil.save(save_path)
        print(f"[log_image_local] Saved image to: {save_path}")


def array2grid(x):
    """Convert image batch to grid numpy array (H, W, 3)."""
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x, nrow=nrow, normalize=True, value_range=(0, 1))
    x = x.clamp(0, 1).mul(255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x
