"""
Inference script for ST-Diff model.

Generate breast ultrasound images from masks and BI-RADS text descriptions.
"""

import argparse
import torch
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm

from model.st_diff import STDiffModel
from utils.preprocessing import preprocess_mask, preprocess_image


def generate_from_mask_and_text(
    model: STDiffModel,
    mask_path: str,
    text_prompt: str,
    output_path: str,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    height: int = 512,
    width: int = 512
):
    """
    Generate image from mask and text prompt.
    
    Args:
        model: Loaded ST-Diff model
        mask_path: Path to input mask image
        text_prompt: BI-RADS text description
        output_path: Path to save generated image
        num_inference_steps: Number of diffusion steps
        guidance_scale: Classifier-free guidance scale
        height: Output image height
        width: Output image width
    """
    # Load and preprocess mask
    mask = Image.open(mask_path).convert("L")
    mask_tensor = preprocess_mask(mask, height, width).unsqueeze(0).to(model.device)
    
    # Generate image
    print(f"Generating image from mask: {mask_path}")
    print(f"Text prompt: {text_prompt}")
    
    generated_images = model.generate(
        masks=mask_tensor,
        text_prompts=[text_prompt],
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width
    )
    
    # Save generated image
    generated_image = Image.fromarray(generated_images[0])
    generated_image.save(output_path)
    print(f"Saved generated image to: {output_path}")


def batch_generate(
    model: STDiffModel,
    mask_dir: str,
    text_file: str,
    output_dir: str,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5
):
    """
    Batch generate images from multiple masks and texts.
    
    Args:
        model: Loaded ST-Diff model
        mask_dir: Directory containing mask images
        text_file: Text file with prompts (one per line, format: mask_name|prompt)
        output_dir: Directory to save generated images
        num_inference_steps: Number of diffusion steps
        guidance_scale: Classifier-free guidance scale
    """
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load text prompts
    with open(text_file, "r") as f:
        lines = f.readlines()
    
    for line in tqdm(lines, desc="Generating images"):
        line = line.strip()
        if not line:
            continue
        
        mask_name, text_prompt = line.split("|", 1)
        mask_path = mask_dir / mask_name
        
        if not mask_path.exists():
            print(f"Warning: Mask not found: {mask_path}")
            continue
        
        output_path = output_dir / f"generated_{mask_path.stem}.png"
        
        generate_from_mask_and_text(
            model=model,
            mask_path=str(mask_path),
            text_prompt=text_prompt,
            output_path=str(output_path),
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )


def main():
    parser = argparse.ArgumentParser(description="Generate images with ST-Diff")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        default=None,
        help="Path to input mask (single image mode)"
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="BI-RADS text prompt (single image mode)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output/generated.png",
        help="Path to save generated image"
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default=None,
        help="Directory containing masks (batch mode)"
    )
    parser.add_argument(
        "--text_file",
        type=str,
        default=None,
        help="Text file with prompts (batch mode, format: mask_name|prompt)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/generated",
        help="Directory to save generated images (batch mode)"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of diffusion steps"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Output image height"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Output image width"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = STDiffModel(device=args.device)
    
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.controlnet.load_state_dict(checkpoint["controlnet_state_dict"])
    if "lora_state_dict" in checkpoint:
        model.pipe.text_encoder.get_peft_model().load_state_dict(
            checkpoint["lora_state_dict"]
        )
    
    model.eval()
    
    # Generate
    if args.mask_path and args.text:
        # Single image mode
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        generate_from_mask_and_text(
            model=model,
            mask_path=args.mask_path,
            text_prompt=args.text,
            output_path=args.output_path,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width
        )
    elif args.mask_dir and args.text_file:
        # Batch mode
        batch_generate(
            model=model,
            mask_dir=args.mask_dir,
            text_file=args.text_file,
            output_dir=args.output_dir,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale
        )
    else:
        parser.error("Either (--mask_path and --text) or (--mask_dir and --text_file) must be provided")


if __name__ == "__main__":
    main()

