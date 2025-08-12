#!/usr/bin/env python3
"""
Cog predictor for Qwen Image generation.
Based on the successful base model inference approach from COMPLETE_QWEN_INFERENCE_GUIDE.md
"""

import os
import sys
import torch
from typing import Optional
from cog import BasePredictor, Input, Path
from PIL import Image

# Make toolkit importable
sys.path.insert(0, "./ai-toolkit")

from extensions_built_in.diffusion_models.qwen_image import QwenImageModel
from toolkit.config_modules import ModelConfig


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.bfloat16
        
        print("Loading Qwen Image model...")
        
        # Load base Qwen model (no LoRA as per successful guide approach)
        model_cfg = ModelConfig(
            name_or_path="Qwen/Qwen-Image", 
            arch="qwen_image", 
            dtype="bf16"
        )
        
        self.qwen = QwenImageModel(
            device=self.device, 
            model_config=model_cfg, 
            dtype=self.torch_dtype
        )
        self.qwen.load_model()
        
        # Build generation pipeline using base model only
        print("Building generation pipeline...")
        self.pipe = self.qwen.get_generation_pipeline()
        
        print("Model loaded successfully!")

    def predict(
        self,
        prompt: str = Input(
            description="Text prompt for image generation",
            default="A beautiful sunset over mountains"
        ),
        width: int = Input(
            description="Width of generated image",
            default=512,
            ge=256,
            le=1024
        ),
        height: int = Input(
            description="Height of generated image", 
            default=512,
            ge=256,
            le=1024
        ),
        guidance_scale: float = Input(
            description="Classifier-free guidance scale. Higher values follow prompt more closely",
            default=7.5,
            ge=1.0,
            le=20.0
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps. More steps = higher quality but slower",
            default=20,
            ge=10,
            le=50
        ),
        seed: Optional[int] = Input(
            description="Random seed for reproducible results. Leave blank for random seed",
            default=None
        )
    ) -> Path:
        """Run image generation prediction"""
        
        print(f"Generating image with prompt: '{prompt}'")
        print(f"Dimensions: {width}x{height}")
        print(f"Guidance scale: {guidance_scale}")
        print(f"Inference steps: {num_inference_steps}")
        
        # Set random seed if provided
        if seed is None:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        print(f"Using seed: {seed}")
        
        # Create generation configuration
        gen_cfg = type("Gen", (), {
            "width": width,
            "height": height, 
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "latents": None,
            "ctrl_img": None
        })()
        
        # Set up generator with seed
        generator = torch.Generator(device=self.qwen.device_torch).manual_seed(seed)
        
        # Get prompt embeddings (conditional and unconditional for CFG)
        print("Encoding prompt...")
        cond = self.qwen.get_prompt_embeds(prompt)
        uncond = self.qwen.get_prompt_embeds("")  # Empty prompt for unconditional
        
        # Generate image
        print("Generating image...")
        img = self.qwen.generate_single_image(
            self.pipe, 
            gen_cfg, 
            cond, 
            uncond, 
            generator, 
            extra={}
        )
        
        # Save output
        output_path = "/tmp/output.png"
        img.save(output_path)
        
        print(f"Image generated successfully and saved to: {output_path}")
        
        return Path(output_path)