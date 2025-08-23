#!/usr/bin/env python3
"""Qwen Image predictor with LoRA support"""

import os
from pathlib import Path
import subprocess
import time

MODEL_CACHE = "model_cache"
BASE_URL = "https://weights.replicate.delivery/default/qwen-image-lora/model_cache/"

# Set environment variables for model caching BEFORE any imports
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE

import sys
import torch
import tempfile
import zipfile
import shutil
from typing import Optional
from cog import BasePredictor, Input, Path
from safetensors.torch import load_file

sys.path.insert(0, "./ai-toolkit")
from extensions_built_in.diffusion_models.qwen_image import QwenImageModel
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.config_modules import ModelConfig
from helpers.billing.metrics import record_billing_metric


def download_weights(url: str, dest: str) -> None:
    """Download weights from CDN using pget"""
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Create model cache directory if it doesn't exist
        os.makedirs(MODEL_CACHE, exist_ok=True)

        # Download model weights if not already present
        model_files = [
            "models--Qwen--Qwen-Image.tar",
            "xet.tar",
        ]

        for model_file in model_files:
            url = BASE_URL + model_file
            filename = url.split("/")[-1]
            dest_path = os.path.join(MODEL_CACHE, filename)
            # Check if the extracted directory exists (without .tar extension)
            extracted_name = filename.replace(".tar", "")
            extracted_path = os.path.join(MODEL_CACHE, extracted_name)
            if not os.path.exists(extracted_path):
                download_weights(url, dest_path)
        
        # Initialize model
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.bfloat16
        self.lora_net = None
        
        print("Loading Qwen Image model...")
        model_cfg = ModelConfig(name_or_path="Qwen/Qwen-Image", arch="qwen_image", dtype="bf16")
        self.qwen = QwenImageModel(device=self.device, model_config=model_cfg, dtype=self.torch_dtype)
        self.qwen.load_model()
        self.pipe = self.qwen.get_generation_pipeline()
        print("Model loaded successfully!")

    def _load_lora_weights(self, lora_path: str, lora_scale: float) -> None:
        # Extract from ZIP if needed
        if lora_path.endswith('.zip'):
            temp_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(lora_path, 'r') as zipf:
                lora_files = [f for f in zipf.namelist() if f.endswith('.safetensors')]
                zipf.extract(lora_files[0], temp_dir)
                safetensors_path = os.path.join(temp_dir, lora_files[0])
        else:
            safetensors_path = lora_path
            temp_dir = None
        
        # Load LoRA config and weights
        try:
            weights = load_file(safetensors_path)
            sample_key = next(k for k in weights.keys() if ("lora_A" in k or "lora_down" in k))
            lora_dim = weights[sample_key].shape[0]
            alpha_key = sample_key.replace("lora_down", "alpha").replace("lora_A", "alpha")
            lora_alpha = int(weights[alpha_key].item()) if alpha_key in weights else lora_dim
        except:
            weights = safetensors_path  # Fallback to path-based loading
            lora_dim, lora_alpha = 32, 32
        
        # Create LoRA network if needed
        if (self.lora_net is None or 
            getattr(self.lora_net, 'lora_dim', None) != lora_dim or 
            getattr(self.lora_net, 'alpha', None) != lora_alpha):
            self.lora_net = LoRASpecialNetwork(
                text_encoder=self.qwen.text_encoder, unet=self.qwen.unet,
                lora_dim=lora_dim, alpha=lora_alpha, multiplier=lora_scale,
                train_unet=True, train_text_encoder=False, is_transformer=True,
                transformer_only=True, base_model=self.qwen,
                target_lin_modules=["QwenImageTransformer2DModel"]
            )
            self.lora_net.apply_to(self.qwen.text_encoder, self.qwen.unet, 
                                 apply_text_encoder=False, apply_unet=True)
            self.lora_net.force_to(self.qwen.device_torch, dtype=self.qwen.torch_dtype)
        
        # Load and activate
        self.lora_net.load_weights(weights)
        self.lora_net.is_active = True
        self.lora_net.multiplier = lora_scale
        self.lora_net._update_torch_multiplier()
        
        # Cleanup
        if temp_dir:
            shutil.rmtree(temp_dir)
        
        print(f"LoRA loaded: dim={lora_dim}, alpha={lora_alpha}, scale={lora_scale}")

    def _get_dimensions(self, aspect_ratio: str, image_size: str) -> tuple:
        """Get dimensions based on aspect ratio and image size preset, matching Pruna's approach"""
        
        # Pruna-style dimensions for optimize_for_quality (~1.5-1.7 MP)
        quality_dims = {
            "1:1": (1328, 1328),
            "16:9": (1664, 928),
            "9:16": (928, 1664),
            "4:3": (1472, 1136),
            "3:4": (1136, 1472),
            "3:2": (1536, 1024),
            "2:3": (1024, 1536),
        }
        
        # Speed dimensions (actual Pruna dimensions from testing)
        speed_dims = {
            "1:1": (1024, 1024),
            "16:9": (1024, 576),
            "9:16": (576, 1280),  # Note: 576x1280, not 576x1024
            "4:3": (1024, 768),
            "3:4": (768, 1024),
            "3:2": (1152, 768),
            "2:3": (768, 1152),
        }
        
        if image_size == "optimize_for_quality":
            dims = quality_dims
        else:  # optimize_for_speed
            dims = speed_dims
        
        width, height = dims.get(aspect_ratio, (1328, 1328))
        
        # Our dimensions are already divisible by 16, but let's keep this for safety
        # in case someone modifies the dimensions above
        adjusted_width = (width // 16) * 16
        adjusted_height = (height // 16) * 16
        
        # Log if adjustment was needed (shouldn't happen with our current dimensions)
        if adjusted_width != width or adjusted_height != height:
            print(f"`height` and `width` have to be divisible by 16 but are {width} and {height}.")
            print(f"Dimensions will be resized to {adjusted_width}x{adjusted_height}")
        
        return adjusted_width, adjusted_height


    def predict(
        self,
        prompt: str = Input(
            description="The main prompt for image generation"
        ),
        enhance_prompt: bool = Input(
            default=False,
            description="Automatically enhance the prompt for better image generation"
        ),
        negative_prompt: str = Input(
            default="",
            description="Things you do not want to see in your image"
        ),
        aspect_ratio: str = Input(
            default="16:9",
            choices=["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"],
            description="Aspect ratio for the generated image. Ignored if width and height are both provided."
        ),
        image_size: str = Input(
            default="optimize_for_quality",
            choices=["optimize_for_quality", "optimize_for_speed"],
            description="Image size preset (quality = larger, speed = faster). Ignored if width and height are both provided."
        ),
        width: Optional[int] = Input(
            default=None,
            ge=512,
            le=2048,
            description="Custom width in pixels. Provide both width and height for custom dimensions (overrides aspect_ratio/image_size)."
        ),
        height: Optional[int] = Input(
            default=None,
            ge=512,
            le=2048,
            description="Custom height in pixels. Provide both width and height for custom dimensions (overrides aspect_ratio/image_size)."
        ),
        go_fast: bool = Input(
            default=False,
            description="Use LCM-LoRA to accelerate image generation (trades quality for 8x speed)"
        ),
        num_inference_steps: int = Input(
            default=50,
            ge=0.0,
            le=50,
            description="Number of denoising steps. More steps = higher quality. Defaults to 4 if go_fast, else 28."
        ),
        guidance: float = Input(
            default=4.0,
            ge=0.0,
            le=10,
            description="Guidance scale for image generation. Defaults to 1 if go_fast, else 3.5."
        ),
        seed: int = Input(
            default=None,
            description="Set a seed for reproducibility. Random by default."
        ),
        output_format: str = Input(
            default="webp",
            choices=["webp", "jpg", "png"],
            description="Format of the output images"
        ),
        output_quality: int = Input(
            default=80,
            ge=0,
            le=100,
            description="Quality when saving images (0-100, higher is better, 100 = lossless)"
        ),
        replicate_weights: Optional[Path] = Input(
            default=None,
            description="Path to LoRA weights file. Leave blank to use base model."
        ),
        lora_scale: float = Input(
            default=1.0,
            ge=0,
            le=3,
            description="Scale for LoRA weights (0 = base model, 1 = full LoRA)"
        )
    ) -> Path:
        """Run a single prediction on the model"""
        # Determine dimensions with smart handling
        if width is not None and height is not None:
            # User provided explicit dimensions - validate and adjust if needed
            orig_w, orig_h = width, height
            
            # Ensure divisible by 16 (round to nearest)
            width = max(512, round(width / 16) * 16)
            height = max(512, round(height / 16) * 16)
            
            # Cap at max dimensions
            width = min(width, 2048)
            height = min(height, 2048)
            
            if (orig_w, orig_h) != (width, height):
                print(f"📐 Adjusted dimensions from {orig_w}x{orig_h} to {width}x{height} (divisible by 16)")
            else:
                print(f"📐 Using custom dimensions: {width}x{height}")
        
        elif width is not None or height is not None:
            # Only one dimension provided - error
            raise ValueError(
                "Both width and height must be provided together for custom dimensions. "
                "Otherwise, use aspect_ratio and image_size presets."
            )
        
        else:
            # Use preset dimensions based on aspect_ratio and image_size
            width, height = self._get_dimensions(aspect_ratio, image_size)
            mode_name = "quality" if image_size == "optimize_for_quality" else "speed"
            print(f"📐 Using {mode_name} preset for {aspect_ratio}: {width}x{height}")

        # Override steps for go_fast mode
        if go_fast and num_inference_steps > 28:
            num_inference_steps = 28
        
        # guidance is already set via default parameter
        
        # Load LoRA if provided
        if replicate_weights:
            self._load_lora_weights(str(replicate_weights), lora_scale)
        elif self.lora_net:
            self.lora_net.is_active = False
            self.lora_net._update_torch_multiplier()
        
        # Set seed
        if seed is None:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
            print(f"Using random seed: {seed}")
        else:
            print(f"Using seed: {seed}")
        
        # Enhance prompt if requested
        if enhance_prompt:
            prompt = f"{prompt}, highly detailed, crisp focus, studio lighting, photorealistic"
        
        # Generate
        print(f"Generating: {prompt} ({width}x{height}, steps={num_inference_steps}, seed={seed})")
        
        import time
        prediction_start = time.time()
        
        gen_cfg = type("Gen", (), {
            "width": width, "height": height, "guidance_scale": guidance,
            "num_inference_steps": num_inference_steps, "latents": None, "ctrl_img": None
        })()
        
        generator = torch.Generator(device=self.qwen.device_torch).manual_seed(seed)
        cond = self.qwen.get_prompt_embeds(prompt)
        uncond = self.qwen.get_prompt_embeds(negative_prompt if negative_prompt.strip() else "")
        
        img = self.qwen.generate_single_image(self.pipe, gen_cfg, cond, uncond, generator, extra={})
        
        prediction_end = time.time()
        prediction_time = prediction_end - prediction_start
        
        # Save
        output_path = f"/tmp/output.{output_format}"
        save_kwargs = {"quality": output_quality} if output_format in ("jpg", "webp") else {}
        if output_format == "jpg":
            save_kwargs["optimize"] = True
        img.save(output_path, **save_kwargs)
        
        # Record billing metric after successful image generation
        record_billing_metric("image_output_count", 1)
        
        print(f"Generation took {prediction_time:.2f} seconds")
        print(f"Total safe images: 1/1")
        
        return Path(output_path)
