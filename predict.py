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
from safetensors.torch import load_file

# Make toolkit importable
sys.path.insert(0, "./ai-toolkit")

from extensions_built_in.diffusion_models.qwen_image import QwenImageModel
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.config_modules import ModelConfig


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.bfloat16
        self.lora_net = None  # Will be set when LoRA weights are loaded
        
        print("Loading Qwen Image model...")
        
        # Load base Qwen model
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
        
        # Build generation pipeline using base model
        print("Building generation pipeline...")
        self.pipe = self.qwen.get_generation_pipeline()
        
        print("Model loaded successfully!")

    def _load_lora_weights(self, lora_path: str, multiplier: float = 1.0) -> None:
        """Load and apply LoRA weights to the model"""
        print(f"Loading LoRA weights from: {lora_path}")
        print(f"LoRA strength multiplier: {multiplier}")
        
        # Load the safetensors file to detect LoRA configuration
        sd = load_file(lora_path)
        
        # Detect LoRA rank/alpha from the state dict
        sample_key = next(k for k in sd.keys() if ("lora_A" in k or "lora_down" in k))
        lora_dim = sd[sample_key].shape[0]
        alpha_key = sample_key.replace("lora_down", "alpha").replace("lora_A", "alpha")
        lora_alpha = int(sd[alpha_key].item()) if alpha_key in sd else lora_dim
        
        print(f"Detected LoRA config - dim: {lora_dim}, alpha: {lora_alpha}")
        
        # Create LoRA network if not already created or if config changed
        if (self.lora_net is None or 
            self.lora_net.lora_dim != lora_dim or 
            self.lora_net.alpha != lora_alpha):
            
            print("Creating new LoRA network...")
            self.lora_net = LoRASpecialNetwork(
                text_encoder=self.qwen.text_encoder,
                unet=self.qwen.unet,  # alias to the Qwen transformer
                lora_dim=lora_dim,
                alpha=lora_alpha,
                multiplier=multiplier,
                train_unet=True,
                train_text_encoder=False,
                is_transformer=True,
                transformer_only=True,
                base_model=self.qwen,
                # Qwen uses QwenImageTransformer2DModel as the target module class
                target_lin_modules=["QwenImageTransformer2DModel"]
            )
            
            # Apply LoRA to the model
            self.lora_net.apply_to(
                self.qwen.text_encoder, 
                self.qwen.unet, 
                apply_text_encoder=False, 
                apply_unet=True
            )
            self.lora_net.force_to(self.qwen.device_torch, dtype=self.qwen.torch_dtype)
            self.lora_net.eval()
        
        # Load the weights and activate
        self.lora_net.load_weights(lora_path)
        self.lora_net.is_active = True
        self.lora_net.multiplier = multiplier  # Set the actual multiplier
        self.lora_net._update_torch_multiplier()
        
        print("LoRA weights loaded and activated successfully!")

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
        ),
        replicate_weights: Optional[Path] = Input(
            description="Path to LoRA weights file (.safetensors). Leave blank to use base model only",
            default=None
        ),
        lora_strength: float = Input(
            description="LoRA strength/multiplier. Higher values = stronger LoRA effect",
            default=1.0,
            ge=0.0,
            le=3.0
        )
    ) -> Path:
        """Run image generation prediction"""
        
        print(f"Generating image with prompt: '{prompt}'")
        print(f"Dimensions: {width}x{height}")
        print(f"Guidance scale: {guidance_scale}")
        print(f"Inference steps: {num_inference_steps}")
        
        # Handle LoRA weights if provided
        if replicate_weights is not None:
            self._load_lora_weights(str(replicate_weights), lora_strength)
        elif self.lora_net is not None:
            # Deactivate any previously loaded LoRA
            self.lora_net.is_active = False
            self.lora_net._update_torch_multiplier()
        
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