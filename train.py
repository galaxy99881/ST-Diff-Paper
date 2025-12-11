"""
Training script for ST-Diff model.

Supports two-stage training:
1. Stage 1: Unconditional pretraining on unlabeled images
2. Stage 2: Multi-condition fine-tuning on labeled triplets (image-mask-text)
"""

import argparse
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb
from pathlib import Path

from model.st_diff import STDiffModel
from data.dataset import UnlabeledDataset, LabeledDataset
from utils.metrics import compute_fid, compute_ssim, compute_lpips
from utils.logging import setup_logger


def train_pretrain_stage(args, config):
    """Stage 1: Unconditional pretraining on unlabeled images."""
    logger = setup_logger("pretrain", args.log_dir)
    logger.info("Starting Stage 1: Unconditional Pretraining")
    
    # Initialize model
    model = STDiffModel(
        base_model_name=config["base_model"],
        device=args.device
    )
    model.train()
    
    # Dataset and dataloader
    dataset = UnlabeledDataset(
        data_dir=config["data"]["unlabeled_dir"],
        image_size=config["data"]["image_size"]
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"]
    )
    
    # Optimizer (only ControlNet parameters)
    optimizer = AdamW(
        list(model.controlnet.parameters()),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"]
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["epochs"] * len(dataloader),
        eta_min=config["training"]["lr"] * 0.01
    )
    
    # Training loop
    global_step = 0
    for epoch in range(config["training"]["epochs"]):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")
        for batch_idx, images in enumerate(pbar):
            images = images.to(args.device)
            
            # Forward pass (unconditional)
            # Note: For unconditional pretraining, we use empty text prompts
            batch_size = images.shape[0]
            empty_prompts = [""] * batch_size
            
            # Create dummy masks (all zeros for unconditional)
            masks = torch.zeros(
                batch_size, 1,
                images.shape[2], images.shape[3],
                device=args.device
            )
            
            loss = model(images, masks, empty_prompts)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.controlnet.parameters(),
                config["training"]["max_grad_norm"]
            )
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # Logging
            if global_step % config["logging"]["log_interval"] == 0:
                logger.info(
                    f"Step {global_step}: Loss={loss.item():.4f}, "
                    f"LR={scheduler.get_last_lr()[0]:.6f}"
                )
                if args.use_wandb:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/epoch": epoch
                    })
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Save checkpoint
        if (epoch + 1) % config["training"]["save_interval"] == 0:
            checkpoint_path = Path(args.checkpoint_dir) / f"pretrain_epoch_{epoch+1}.ckpt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.controlnet.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": epoch_loss / len(dataloader)
            }, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    logger.info("Pretraining completed!")


def train_finetune_stage(args, config):
    """Stage 2: Multi-condition fine-tuning on labeled triplets."""
    logger = setup_logger("finetune", args.log_dir)
    logger.info("Starting Stage 2: Multi-Condition Fine-tuning")
    
    # Initialize model
    model = STDiffModel(
        base_model_name=config["base_model"],
        device=args.device
    )
    
    # Load pretrained ControlNet if specified
    if args.pretrained_path:
        logger.info(f"Loading pretrained model from {args.pretrained_path}")
        checkpoint = torch.load(args.pretrained_path, map_location=args.device)
        model.controlnet.load_state_dict(checkpoint["model_state_dict"])
    
    model.train()
    
    # Dataset and dataloader
    dataset = LabeledDataset(
        image_dir=config["data"]["labeled_dir"] / "images",
        mask_dir=config["data"]["labeled_dir"] / "masks",
        text_file=config["data"]["labeled_dir"] / "texts.txt",
        image_size=config["data"]["image_size"]
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"]
    )
    
    # Optimizer (ControlNet + LoRA parameters)
    trainable_params = (
        list(model.controlnet.parameters()) +
        list(model.pipe.text_encoder.get_peft_model().parameters())
    )
    
    optimizer = AdamW(
        trainable_params,
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"]
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["epochs"] * len(dataloader),
        eta_min=config["training"]["lr"] * 0.01
    )
    
    # Training loop
    global_step = 0
    best_loss = float("inf")
    
    for epoch in range(config["training"]["epochs"]):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")
        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(args.device)
            masks = batch["mask"].to(args.device)
            texts = batch["text"]
            
            # Forward pass
            loss = model(images, masks, texts)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainable_params,
                config["training"]["max_grad_norm"]
            )
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # Logging
            if global_step % config["logging"]["log_interval"] == 0:
                logger.info(
                    f"Step {global_step}: Loss={loss.item():.4f}, "
                    f"LR={scheduler.get_last_lr()[0]:.6f}"
                )
                if args.use_wandb:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/epoch": epoch
                    })
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / len(dataloader)
        
        # Save checkpoint
        if (epoch + 1) % config["training"]["save_interval"] == 0:
            checkpoint_path = Path(args.checkpoint_dir) / f"finetune_epoch_{epoch+1}.ckpt"
            torch.save({
                "epoch": epoch,
                "controlnet_state_dict": model.controlnet.state_dict(),
                "lora_state_dict": model.pipe.text_encoder.get_peft_model().state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss
            }, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = Path(args.checkpoint_dir) / "best_model.ckpt"
            torch.save({
                "epoch": epoch,
                "controlnet_state_dict": model.controlnet.state_dict(),
                "lora_state_dict": model.pipe.text_encoder.get_peft_model().state_dict(),
                "loss": avg_loss
            }, best_path)
            logger.info(f"Saved best model (loss={avg_loss:.4f}): {best_path}")
    
    logger.info("Fine-tuning completed!")


def main():
    parser = argparse.ArgumentParser(description="Train ST-Diff model")
    parser.add_argument(
        "--stage",
        type=str,
        choices=["pretrain", "finetune"],
        required=True,
        help="Training stage: pretrain or finetune"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=None,
        help="Path to pretrained checkpoint (for finetune stage)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory to save logs"
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use Weights & Biases for logging"
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project="st-diff",
            name=f"{args.stage}_{config.get('experiment_name', 'default')}",
            config=config
        )
    
    # Create directories
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    
    # Run training
    if args.stage == "pretrain":
        train_pretrain_stage(args, config)
    else:
        train_finetune_stage(args, config)
    
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

