# Source:
#   RAE: https://github.com/bytetriper/RAE/blob/main/src/utils/model_utils.py
#        https://github.com/bytetriper/RAE/blob/main/src/utils/train_utils.py


from __future__ import annotations
import importlib
from dataclasses import dataclass
from typing import Union, Tuple, Optional, List
import torch.nn as nn
from omegaconf import OmegaConf
import torch
from omegaconf import OmegaConf, DictConfig



def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config) -> object:
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    model = get_obj_from_str(config["target"])(**config.get("params", dict()))
    ckpt_path = config.get("ckpt", None)
    if ckpt_path is not None:
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if "ema" in state_dict:
            state_dict = state_dict["ema"]
        elif "model" in state_dict:
            raise NotImplementedError("Loading from 'model' key not implemented yet.")
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=True)
        print(f'target {config["target"]} loaded from {ckpt_path}')
    return model



def parse_configs(config_path: str) -> Tuple[DictConfig, DictConfig, DictConfig, DictConfig, DictConfig, DictConfig, DictConfig]:
    """Load a config file and return component sections as DictConfigs."""
    config = OmegaConf.load(config_path)
    dit_config = config.get("model", None)
    transport_config = config.get("transport", None)
    sampler_config = config.get("sampler", None)
    guidance_config = config.get("guidance", None)
    misc = config.get("misc", None)
    training_config = config.get("training", None)
    exp_name = config.get("exp_name", None)
    return dit_config, transport_config, sampler_config, guidance_config, misc, training_config, exp_name

def none_or_str(value):
    if value == 'None':
        return None
    return value