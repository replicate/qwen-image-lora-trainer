#!/usr/bin/env python3
"""Qwen Image predictor with LoRA support"""

import os
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


    def predict(
        self,
        prompt: str = Input(description="Prompt for generated image", default="A beautiful sunset over mountains"),
        enhance_prompt: bool = Input(description="Enhance the prompt with positive magic", default=False),
        negative_prompt: str = Input(description="Negative prompt for generated image", default=""),
        aspect_ratio: str = Input(description="Aspect ratio", choices=["1:1", "16:9", "9:16", "4:3", "3:4"], default="16:9"),
        image_size: str = Input(description="Image size preset", choices=["optimize_for_quality", "optimize_for_speed"], default="optimize_for_quality"),
        width: int = Input(description="Width (overrides aspect_ratio/image_size)", default=512, ge=256, le=1024),
        height: int = Input(description="Height (overrides aspect_ratio/image_size)", default=512, ge=256, le=1024),
        go_fast: bool = Input(description="Run faster with minor optimizations", default=False),
        num_inference_steps: int = Input(description="Number of denoising steps", default=50, ge=1, le=50),
        guidance: float = Input(description="Guidance scale", default=4.0, ge=0.0, le=10.0),
        seed: Optional[int] = Input(description="Random seed (leave blank for random)", default=None),
        output_format: str = Input(description="Output format", choices=["webp", "jpg", "png"], default="webp"),
        output_quality: int = Input(description="Quality for jpg/webp", default=80, ge=0, le=100),
        replicate_weights: Optional[Path] = Input(description="LoRA weights (.safetensors or .zip)", default=None),
        lora_scale: float = Input(description="LoRA strength", default=1.0, ge=0.0, le=3.0)
    ) -> Path:
        """Generate an image based on the provided prompt and parameters."""
        # Record billing metric at the start (before any processing that might fail)
        record_billing_metric("image_output_count", 1)
        
        # Fast mode optimization
        if go_fast and num_inference_steps > 28:
            num_inference_steps = 28
        
        # Load LoRA if provided
        if replicate_weights:
            self._load_lora_weights(str(replicate_weights), lora_scale)
        elif self.lora_net:
            self.lora_net.is_active = False
            self.lora_net._update_torch_multiplier()
        
        # Set seed
        if seed is None:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        
        # Calculate dimensions
        if width == 512 and height == 512:
            long_side = 1024 if image_size == "optimize_for_quality" else 512
            if aspect_ratio == "1:1":
                width, height = long_side, long_side
            elif aspect_ratio == "16:9":
                width, height = long_side, int(long_side * 9 / 16)
            elif aspect_ratio == "9:16":
                width, height = int(long_side * 9 / 16), long_side
            elif aspect_ratio == "4:3":
                width, height = long_side, int(long_side * 3 / 4)
            elif aspect_ratio == "3:4":
                width, height = int(long_side * 3 / 4), long_side
            width = max(256, min(1024, width))
            height = max(256, min(1024, height))
        
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
        
        print(f"PREDICTION_TIME: {prediction_time:.2f} seconds")
        print(f"Generated: {output_path}")
        return Path(output_path)
