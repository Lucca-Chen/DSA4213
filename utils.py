
"""General Utility Functions"""
import json
import torch
from pathlib import Path

def load_config(config_path="configs/project_config.json"):
    """Load project configuration"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_model_size(model):
    """Calculate model parameter size"""
    param_size = 0
    trainable_params = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        if param.requires_grad:
            trainable_params += param.nelement()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size = (param_size + buffer_size) / 1024**2
    total_params = sum(p.nelement() for p in model.parameters())
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "size_mb": total_size
    }

def print_gpu_memory():
    """Print GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    else:
        print("GPU not in use")
