#!/usr/bin/env python3
"""
LoRA Inference for Qwen-Image on Replicate

Simple inference package that loads base Qwen-Image model + LoRA adapters
and generates images. Supports optional placeholder token registration.

Usage:
    python predict.py \
        --prompt "a photo of <zwx>" \
        --lora_path "./output/lora_weights.safetensors" \
        --placeholder_token "<zwx>" \
        --output_path "./generated_image.jpg"
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Union
import torch
from PIL import Image
import safetensors.torch
from diffusers import QwenImagePipeline
from transformers import Qwen2Tokenizer, Qwen2_5_VLForConditionalGeneration
from diffusers.loaders import LoraLoaderMixin


class QwenImageLoRAInference:
    """LoRA inference for Qwen-Image with optional placeholder token support"""
    
    def __init__(
        self,
        base_model_path: str = "Qwen/Qwen-Image",
        device: str = "cuda",
        dtype: str = "bf16",
    ):
        self.base_model_path = base_model_path
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        
        self.pipeline = None
        self.tokenizer = None
        self.text_encoder = None
        self.placeholder_token_id = None
        
    def load_base_model(self):
        """Load the base Qwen-Image model"""
        print(f"Loading base model from {self.base_model_path}...")
        
        # Load pipeline
        self.pipeline = QwenImagePipeline.from_pretrained(
            self.base_model_path,
            torch_dtype=self.dtype,
            device_map=self.device,
        )
        
        # Get references for convenience
        self.tokenizer = self.pipeline.tokenizer
        self.text_encoder = self.pipeline.text_encoder
        
        print("✓ Base model loaded successfully")
        
    def register_placeholder_token(self, placeholder_token: str):
        """Register a placeholder token (e.g., '<zwx>') for inference"""
        if not placeholder_token:
            return
            
        print(f"Registering placeholder token: {placeholder_token}")
        
        # Add the special token to the tokenizer
        num_added_tokens = self.tokenizer.add_special_tokens({
            "additional_special_tokens": [placeholder_token]
        })
        
        if num_added_tokens == 0:
            print(f"ℹ Token {placeholder_token} already exists in tokenizer")
        else:
            # Resize text encoder embeddings to accommodate new token
            self.text_encoder.resize_token_embeddings(len(self.tokenizer))
            print(f"✓ Added new token, vocabulary size: {len(self.tokenizer)}")
        
        # Get the token ID
        self.placeholder_token_id = self.tokenizer.convert_tokens_to_ids(placeholder_token)
        
        # Check if token encodes to single ID
        test_ids = self.tokenizer.encode(placeholder_token, add_special_tokens=False)
        if len(test_ids) != 1:
            print(f"⚠ WARNING: Token {placeholder_token} encodes to {len(test_ids)} tokens: {test_ids}")
            print("Consider using angle brackets like <zwx> to ensure single token encoding")
        
        print(f"✓ Placeholder token '{placeholder_token}' registered with ID {self.placeholder_token_id}")
        
    def load_lora_weights(self, lora_path: str, lora_scale: float = 1.0):
        """Load LoRA weights into the model"""
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA weights not found: {lora_path}")
            
        print(f"Loading LoRA weights from {lora_path}...")
        
        # Load LoRA using diffusers built-in loader
        if lora_path.endswith('.safetensors'):
            self.pipeline.load_lora_weights(lora_path)
        else:
            # Assume it's a directory with safetensors files
            self.pipeline.load_lora_weights(lora_path)
            
        # Set LoRA scale
        if hasattr(self.pipeline, 'set_adapters'):
            self.pipeline.set_adapters(["default"], adapter_weights=[lora_scale])
        
        print(f"✓ LoRA weights loaded with scale {lora_scale}")
        
    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 20,
        guidance_scale: float = 4.0,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Generate an image from a text prompt"""
        
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call load_base_model() first.")
            
        # Set seed for reproducibility
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
            
        print(f"Generating image: '{prompt}'")
        print(f"Settings: {width}x{height}, steps={num_inference_steps}, guidance={guidance_scale}")
        
        # Generate image
        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            
        image = result.images[0]
        print("✓ Image generated successfully")
        
        return image


def main():
    parser = argparse.ArgumentParser(description="Qwen-Image LoRA Inference")
    parser.add_argument("--prompt", required=True, help="Text prompt for image generation")
    parser.add_argument("--negative_prompt", default="", help="Negative prompt")
    parser.add_argument("--lora_path", help="Path to LoRA weights (.safetensors file or directory)")
    parser.add_argument("--lora_scale", type=float, default=1.0, help="LoRA scale factor")
    parser.add_argument("--placeholder_token", help="Placeholder token (e.g., '<zwx>')")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=4.0, help="Guidance scale")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--output_path", default="generated_image.jpg", help="Output image path")
    parser.add_argument("--base_model", default="Qwen/Qwen-Image", help="Base model path/name")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--dtype", default="bf16", help="Data type (bf16/fp16/fp32)")
    
    args = parser.parse_args()
    
    try:
        # Initialize inference
        inference = QwenImageLoRAInference(
            base_model_path=args.base_model,
            device=args.device,
            dtype=args.dtype,
        )
        
        # Load base model
        inference.load_base_model()
        
        # Register placeholder token if provided
        if args.placeholder_token:
            inference.register_placeholder_token(args.placeholder_token)
            
        # Load LoRA weights if provided
        if args.lora_path:
            inference.load_lora_weights(args.lora_path, args.lora_scale)
            
        # Generate image
        image = inference.generate_image(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            width=args.width,
            height=args.height,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
        )
        
        # Save result
        image.save(args.output_path)
        print(f"✓ Image saved to {args.output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()