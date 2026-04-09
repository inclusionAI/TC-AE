# Copyright (c) 2026 inclusionAI
# SPDX-License-Identifier: MIT

import os
import shutil
import torch
import numpy as np
import random
import logging
import torch.optim as optim
from torch.utils.data import DataLoader
from tcae.models.config import VitConfig
from tcae.models.tcae import TCAE



def seed_everything(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    
    np.random.seed(seed)
    
    random.seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v 
        else:
            new_state_dict[k] = v
    return new_state_dict



def build_model(model_config, patch_nums):
    encoderconfig = VitConfig(**model_config.vitencoder_config)
    decoderconfig = VitConfig(**model_config.vitdecoder_config)
    model = TCAE(z_channels=model_config.z_channels, 
                patch_nums=patch_nums,
                encoderconfig=encoderconfig,
                decoderconfig=decoderconfig,
                decoder_downsample_type=model_config.decoder_downsample_type,
                scale_type=model_config.scale_type,
                use_contrast_patch_size=model_config.use_contrast_patch_size,
                use_vf=model_config.use_vf,
                use_cls_head=model_config.use_cls_head,
                use_mae=model_config.use_mae,
                use_simmim=model_config.use_simmim,
                use_mutual=model_config.use_mutual,
                use_ssl=model_config.use_ssl,
                ssl_cfg=model_config.ssl_cfg,
                reverse_proj=model_config.reverse_proj,
                decoder_attn_type = model_config.decoder_attn_type,
                use_tanh=model_config.use_tanh,
                test_mode=False,)
    return model



def get_image_paths(folder_path):
    valid_image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
    
    image_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_image_extensions:
                image_paths.append(os.path.join(root, file))
    
    return image_paths



def get_image_data(image_path, oss_obj_antdiff, oss_obj_collect, oss_obj_imagenet):
    oss_img_path = get_antdiffdata_oss_image_path(image_path)
    # print(oss_img_path)
    if oss_img_path is None:
        return None
    # Another oss bucket for backup.
    if isinstance(oss_img_path, List):
        ## imagenet
        oss_img_path = oss_img_path[0]
        try:
            vfile_bytes = io.BytesIO(oss_obj_imagenet.open(oss_img_path, 'rb').read())
            img = Image.open(vfile_bytes).convert('RGB')
            return img
        except:
            print('*' * 10, 'Read file failed:', oss_img_path)
            return None
    oss_img_path_collect = None
    if isinstance(oss_img_path, tuple):
        oss_img_path, oss_img_path_collect = oss_img_path
    try:
        vfile_bytes = io.BytesIO(oss_obj_antdiff.open(oss_img_path, 'rb').read())
        img = Image.open(vfile_bytes).convert('RGB')
        # print('antdiff:', img.size)
    except:
        if oss_img_path_collect is not None:  # Choose the backup oss bucket.
            try:
                vfile_bytes = io.BytesIO(oss_obj_collect.open(oss_img_path_collect, 'rb').read())
                img = Image.open(vfile_bytes).convert('RGB')
                # print('collect:', img.size)
            except:
                print('*' * 10, 'Read file failed:', oss_img_path_collect, image_path)
                img = None
        else:
            print('*' * 10, 'Read file failed:', oss_img_path, image_path)
            img = None
    return img
