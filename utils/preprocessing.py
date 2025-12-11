"""
Preprocessing utilities for images and masks.
"""

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


def preprocess_image(image: Image.Image, size: int = 512) -> torch.Tensor:
    """
    Preprocess image for model input.
    
    Args:
        image: PIL Image
        size: Target size
        
    Returns:
        Preprocessed image tensor [C, H, W] in range [-1, 1]
    """
    image = image.convert("RGB")
    image = image.resize((size, size), Image.LANCZOS)
    
    image_array = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
    image_tensor = image_tensor * 2.0 - 1.0
    
    return image_tensor


def preprocess_mask(mask: Image.Image, height: int = 512, width: int = 512) -> torch.Tensor:
    """
    Preprocess mask for model input.
    
    Args:
        mask: PIL Image (grayscale mask)
        height: Target height
        width: Target width
        
    Returns:
        Preprocessed mask tensor [1, H, W] in range [0, 1]
    """
    mask = mask.convert("L")
    mask = mask.resize((height, width), Image.NEAREST)
    
    mask_array = np.array(mask).astype(np.float32) / 255.0
    mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
    
    return mask_tensor


def postprocess_image(image_tensor: torch.Tensor) -> Image.Image:
    """
    Postprocess model output to PIL Image.
    
    Args:
        image_tensor: Image tensor [C, H, W] in range [-1, 1]
        
    Returns:
        PIL Image in range [0, 255]
    """
    # Denormalize from [-1, 1] to [0, 1]
    image_tensor = (image_tensor + 1.0) / 2.0
    image_tensor = torch.clamp(image_tensor, 0, 1)
    
    # Convert to numpy and permute
    image_array = image_tensor.permute(1, 2, 0).cpu().numpy()
    image_array = (image_array * 255).astype(np.uint8)
    
    return Image.fromarray(image_array)

