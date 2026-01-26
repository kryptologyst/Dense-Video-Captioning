"""Utility functions for device management and reproducibility."""

import os
import random
from typing import Optional, Union

import numpy as np
import torch
import torch.backends.cudnn as cudnn


def get_device(device: Optional[str] = None) -> torch.device:
    """Get the best available device.
    
    Args:
        device: Device specification. If None or 'auto', automatically select best device.
        
    Returns:
        torch.device: The selected device.
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    return torch.device(device)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
        deterministic: Whether to use deterministic algorithms.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            cudnn.deterministic = True
            cudnn.benchmark = False
        else:
            cudnn.benchmark = True
    
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    # Set environment variables for additional reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_model_size(model: torch.nn.Module) -> dict:
    """Calculate model size metrics.
    
    Args:
        model: PyTorch model.
        
    Returns:
        dict: Dictionary containing model size metrics.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
        "total_parameters_millions": total_params / 1e6,
        "trainable_parameters_millions": trainable_params / 1e6,
    }


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device_memory_info(device: Union[str, torch.device]) -> dict:
    """Get memory information for the specified device.
    
    Args:
        device: Device to get memory info for.
        
    Returns:
        dict: Memory information dictionary.
    """
    device = torch.device(device) if isinstance(device, str) else device
    
    if device.type == "cuda":
        return {
            "device": str(device),
            "total_memory_gb": torch.cuda.get_device_properties(device).total_memory / 1e9,
            "allocated_memory_gb": torch.cuda.memory_allocated(device) / 1e9,
            "cached_memory_gb": torch.cuda.memory_reserved(device) / 1e9,
            "free_memory_gb": (torch.cuda.get_device_properties(device).total_memory - 
                             torch.cuda.memory_reserved(device)) / 1e9,
        }
    elif device.type == "mps":
        return {
            "device": str(device),
            "total_memory_gb": None,  # MPS doesn't expose this
            "allocated_memory_gb": None,
            "cached_memory_gb": None,
            "free_memory_gb": None,
        }
    else:
        return {
            "device": str(device),
            "total_memory_gb": None,
            "allocated_memory_gb": None,
            "cached_memory_gb": None,
            "free_memory_gb": None,
        }
