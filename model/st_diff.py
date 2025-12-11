"""
ST-Diff: Mask-Conditioned Breast Ultrasound Image Generation Framework

This module implements the core ST-Diff architecture combining:
- Latent Diffusion Model (LDM) as base generator
- ControlNet for structural mask conditioning
- CLIP-LoRA for semantic text conditioning
"""

import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline, ControlNetModel, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np


class STDiffModel(nn.Module):
    """
    ST-Diff: Mask-Conditioned Breast Ultrasound Image Generation Model
    
    Architecture:
    1. Base: Latent Diffusion Model (Stable Diffusion)
    2. Structure Branch: ControlNet for mask conditioning
    3. Semantic Branch: CLIP-LoRA for BI-RADS text conditioning
    """
    
    def __init__(
        self,
        base_model_name: str = "runwayml/stable-diffusion-v1-5",
        controlnet_model_name: str = None,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        device: str = "cuda"
    ):
        """
        Initialize ST-Diff model.
        
        Args:
            base_model_name: HuggingFace model name for base LDM
            controlnet_model_name: Path to ControlNet model (if None, will be created)
            lora_rank: LoRA rank for CLIP adapter
            lora_alpha: LoRA alpha scaling factor
            device: Device to run model on
        """
        super().__init__()
        self.device = device
        
        # Load base Stable Diffusion pipeline
        print(f"Loading base model: {base_model_name}")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        self.pipe = self.pipe.to(device)
        
        # Initialize ControlNet for mask conditioning
        if controlnet_model_name is None:
            print("Initializing ControlNet from base UNet...")
            self.controlnet = ControlNetModel.from_unet(
                self.pipe.unet,
                conditioning_embedding_out_channels=(16, 32, 96, 256)
            )
        else:
            print(f"Loading ControlNet: {controlnet_model_name}")
            self.controlnet = ControlNetModel.from_pretrained(controlnet_model_name)
        
        self.controlnet = self.controlnet.to(device)
        
        # Setup CLIP-LoRA for semantic adaptation
        print("Setting up CLIP-LoRA adapter...")
        self._setup_clip_lora(lora_rank, lora_alpha)
        
        # Freeze base model parameters (only train ControlNet and LoRA)
        self._freeze_base_model()
        
    def _setup_clip_lora(self, rank: int, alpha: int):
        """Setup LoRA adapter for CLIP text encoder."""
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
            task_type=TaskType.FEATURE_EXTRACTION,
            bias="none"
        )
        
        self.pipe.text_encoder = get_peft_model(
            self.pipe.text_encoder,
            lora_config
        )
        
    def _freeze_base_model(self):
        """Freeze base UNet and VAE, only train ControlNet and LoRA."""
        # Freeze UNet (except ControlNet connections)
        for param in self.pipe.unet.parameters():
            param.requires_grad = False
        
        # Freeze VAE
        for param in self.pipe.vae.parameters():
            param.requires_grad = False
        
        # ControlNet and LoRA remain trainable
        print("Base model frozen. Only ControlNet and LoRA are trainable.")
    
    def forward(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        text_prompts: list,
        timesteps: torch.Tensor = None,
        guidance_scale: float = 7.5
    ):
        """
        Forward pass for training.
        
        Args:
            images: Input images [B, C, H, W]
            masks: Segmentation masks [B, 1, H, W]
            text_prompts: List of BI-RADS text descriptions
            timesteps: Diffusion timesteps [B]
            guidance_scale: Classifier-free guidance scale
            
        Returns:
            loss: Training loss
        """
        # Encode images to latent space
        with torch.no_grad():
            latents = self.pipe.vae.encode(images).latent_dist.sample()
            latents = latents * self.pipe.vae.config.scaling_factor
        
        # Encode text prompts
        text_inputs = self.pipe.tokenizer(
            text_prompts,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        text_embeddings = self.pipe.text_encoder(
            text_inputs.input_ids
        )[0]
        
        # Prepare mask conditioning
        # Resize mask to match latent dimensions
        mask_resized = torch.nn.functional.interpolate(
            masks,
            size=(latents.shape[2], latents.shape[3]),
            mode="bilinear",
            align_corners=False
        )
        
        # Sample random timesteps if not provided
        if timesteps is None:
            timesteps = torch.randint(
                0,
                self.pipe.scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=self.device
            ).long()
        
        # Add noise to latents
        noise = torch.randn_like(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timesteps)
        
        # Get ControlNet conditioning
        controlnet_down_block_res_samples, controlnet_mid_block_res_sample = \
            self.controlnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=text_embeddings,
                controlnet_cond=mask_resized,
                return_dict=False
            )
        
        # UNet forward with ControlNet conditioning
        model_pred = self.pipe.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=text_embeddings,
            down_block_additional_residuals=controlnet_down_block_res_samples,
            mid_block_additional_residual_sample=controlnet_mid_block_res_sample
        ).sample
        
        # Compute loss
        if self.pipe.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.pipe.scheduler.config.prediction_type == "v_prediction":
            target = self.pipe.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.pipe.scheduler.config.prediction_type}")
        
        loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")
        
        return loss
    
    @torch.no_grad()
    def generate(
        self,
        masks: torch.Tensor,
        text_prompts: list,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512
    ):
        """
        Generate images from masks and text prompts.
        
        Args:
            masks: Segmentation masks [B, 1, H, W]
            text_prompts: List of BI-RADS text descriptions
            num_inference_steps: Number of diffusion steps
            guidance_scale: Classifier-free guidance scale
            height: Output image height
            width: Output image width
            
        Returns:
            generated_images: Generated images [B, C, H, W]
        """
        # Encode text
        text_inputs = self.pipe.tokenizer(
            text_prompts,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        text_embeddings = self.pipe.text_encoder(text_inputs.input_ids)[0]
        
        # Prepare mask conditioning
        mask_resized = torch.nn.functional.interpolate(
            masks,
            size=(height // 8, width // 8),  # Latent space dimensions
            mode="bilinear",
            align_corners=False
        )
        
        # Initialize latents
        latents = torch.randn(
            (len(text_prompts), 4, height // 8, width // 8),
            device=self.device,
            dtype=text_embeddings.dtype
        )
        
        # Set scheduler timesteps
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.pipe.scheduler.timesteps
        
        # Denoising loop
        for t in timesteps:
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)
            
            # ControlNet conditioning
            controlnet_down_block_res_samples, controlnet_mid_block_res_sample = \
                self.controlnet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=torch.cat([text_embeddings] * 2),
                    controlnet_cond=torch.cat([mask_resized] * 2),
                    return_dict=False
                )
            
            # UNet prediction
            noise_pred = self.pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=torch.cat([text_embeddings] * 2),
                down_block_additional_residuals=controlnet_down_block_res_samples,
                mid_block_additional_residual_sample=controlnet_mid_block_res_sample
            ).sample
            
            # Perform classifier-free guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Update latents
            latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample
        
        # Decode latents to images
        latents = 1 / self.pipe.vae.config.scaling_factor * latents
        images = self.pipe.vae.decode(latents).sample
        
        # Post-process images
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).astype(np.uint8)
        
        return images

