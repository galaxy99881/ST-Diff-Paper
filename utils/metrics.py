"""
Evaluation metrics for ST-Diff model.

Includes FID, SSIM, LPIPS, Mask Dice, and BI-RADS accuracy.
"""

import torch
import numpy as np
from scipy import linalg
from skimage.metrics import structural_similarity as ssim
import lpips
from torchmetrics.functional import dice


def compute_fid(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    """
    Compute Fréchet Inception Distance (FID).
    
    Args:
        real_features: Features from real images [N, D]
        fake_features: Features from fake images [M, D]
        
    Returns:
        FID score
    """
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute Structural Similarity Index (SSIM).
    
    Args:
        img1: First image [H, W, C] or [H, W]
        img2: Second image [H, W, C] or [H, W]
        
    Returns:
        SSIM score
    """
    if img1.ndim == 3:
        return ssim(img1, img2, multichannel=True, channel_axis=2)
    else:
        return ssim(img1, img2)


def compute_lpips(img1: torch.Tensor, img2: torch.Tensor, net: str = "alex") -> float:
    """
    Compute Learned Perceptual Image Patch Similarity (LPIPS).
    
    Args:
        img1: First image tensor [B, C, H, W] in range [0, 1]
        img2: Second image tensor [B, C, H, W] in range [0, 1]
        net: Network to use ('alex', 'vgg', 'squeeze')
        
    Returns:
        LPIPS score
    """
    loss_fn = lpips.LPIPS(net=net)
    with torch.no_grad():
        dist = loss_fn(img1, img2)
    return dist.mean().item()


def compute_mask_dice(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
    """
    Compute Dice coefficient for mask segmentation.
    
    Args:
        pred_mask: Predicted mask [B, 1, H, W] or [B, H, W]
        gt_mask: Ground truth mask [B, 1, H, W] or [B, H, W]
        
    Returns:
        Dice coefficient
    """
    if pred_mask.dim() == 4:
        pred_mask = pred_mask.squeeze(1)
    if gt_mask.dim() == 4:
        gt_mask = gt_mask.squeeze(1)
    
    dice_score = dice(pred_mask, gt_mask)
    return dice_score.item()


def compute_birads_accuracy(pred_labels: list, gt_labels: list) -> float:
    """
    Compute BI-RADS classification accuracy.
    
    Args:
        pred_labels: List of predicted BI-RADS labels
        gt_labels: List of ground truth BI-RADS labels
        
    Returns:
        Accuracy score
    """
    correct = sum(p == g for p, g in zip(pred_labels, gt_labels))
    return correct / len(gt_labels) if len(gt_labels) > 0 else 0.0

