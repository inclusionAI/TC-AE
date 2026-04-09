# Copyright (c) 2026 inclusionAI
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
import PIL.Image as PImage
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import InterpolationMode, transforms
import os
from PIL import Image
from datasets import load_dataset
from torch.utils.data import Dataset
import fsspec
import json
import os
from PIL import Image
from datasets import load_dataset
from torch.utils.data import Dataset
from .utils import get_image_data
import random



def pil_loader(path):
    with open(path, 'rb') as f:
        img: PImage.Image = PImage.open(f).convert('RGB')
    return img


def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)


class ResImageNet(DatasetFolder):
    ## TODO: change dataloader method, read jsonl with label
    def __init__(self, root, transform=None, target_transform=None, loader=pil_loader):
        super(ResImageNet, self).__init__(root, loader, IMG_EXTENSIONS, transform=transform, target_transform=target_transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        raw_sample = self.loader(path)
        
        if self.transform is not None:
            sample = self.transform(raw_sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target



def build_eval_datataset(
    data_path: str, final_reso: int, mid_reso=1.125,
):
    # build augmentations
    mid_reso = round(mid_reso * final_reso)  # first resize to mid_reso, then crop to final_reso
    val_aug = [
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS), # transforms.Resize: resize the shorter edge to mid_reso
        transforms.CenterCrop((final_reso, final_reso)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
    val_aug = transforms.Compose(val_aug)
    
    val_set = ResImageNet(data_path, transform=val_aug)
    
    return val_set