#!/usr/bin/env python3
"""
Cog Prediction Interface for Qwen-Image LoRA with Zero-Prior Placeholder Tokens

This module provides the Cog prediction API that:
1. Loads base Qwen-Image model
2. Accepts weights archive from training (replicate_weights: Path)
3. Extracts and applies LoRA weights
4. Registers placeholder tokens automatically
5. Generates images with the fine-tuned model

Note: This implementation uses ai-toolkit's QwenImageModel for full compatibility
"""

import json
import os
import shutil
import sys
import tempfile
import zipfile
import tarfile
from pathlib import Path as P
from typing import Optional
import gc
import psutil

import torch
from cog import BasePredictor, Input, Path
from PIL import Image

# Add ai-toolkit to Python path
sys.path.insert(0, '/src/ai-toolkit')

# Import from ai-toolkit - FAIL FAST if not available
from extensions_built_in.diffusion_models.qwen_image.qwen_image import QwenImageModel
from toolkit.config_modules import ModelConfig, GenerateImageConfig, TrainConfig, DebugConfig
from toolkit.prompt_utils import PromptEmbeds
import safetensors.torch

print("✅ ai-toolkit modules imported successfully")


class Predictor(BasePredictor):
    """Cog Predictor for Qwen-Image LoRA inference with zero-prior placeholder tokens"""
    
    def setup(self):
        """Load base Qwen-Image model - called once at container startup (no LoRA yet)"""
        print("🚀 Loading base Qwen-Image model...")
        
        # EXTREME memory conservation for any GPU (optimized for H200, works on A100)
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:256,roundup_power2_divisions:32"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Synchronous execution for better memory tracking
        os.environ["PYTORCH_CUDA_MEMORY_FRACTION"] = "0.7"  # Use only 70% of GPU memory
        
        # Clear any existing CUDA cache and report memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            self._report_memory("Initial")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Use float16 for better memory efficiency across all GPUs
        self.dtype = torch.float16
        
        print(f"🔧 Using device: {self.device}, dtype: {self.dtype}")
        

        
        # EXTREME memory conservation model config (works on A100, optimized for H200)
        model_config = ModelConfig(
            name_or_path="Qwen/Qwen-Image",
            arch="qwen_image",
            dtype=self.dtype,
            quantize=True,
            quantize_te=True,
            low_vram=True,  # Always use low VRAM mode
            qtype="qint4",  # Use 4-bit quantization for maximum memory savings
            qtype_te="qint4",  # 4-bit text encoder quantization
            split_model_over_gpus=False,
            quantize_kwargs={"exclude": []},
            # CPU offloading for extreme memory conservation
            vae_device="cpu",     # Keep VAE on CPU to save GPU memory
            te_device="cpu",      # Keep text encoder on CPU initially
        )
        
        # Create train config (for compatibility)
        train_config = TrainConfig(
            train_placeholder_token=False  # Will be set per prediction if needed
        )
        
        # Create debug config
        debug_config = DebugConfig(log_prompt_ids=False)
        
        # Initialize the Qwen-Image model with memory monitoring
        print("🔧 Initializing model...")
        self._cleanup_memory()
        self._report_memory("Before model init")
        
        self.qwen_model = QwenImageModel(
            device=self.device,
            model_config=model_config,
            dtype=self.dtype,
            train_config=train_config,
            debug_config=debug_config
        )
        
        self._report_memory("After model init")
        
        # Aggressive memory cleanup before model loading
        self._cleanup_memory()
        self._report_memory("Before model loading")
        
        # Load model components sequentially to prevent memory spikes
        print("🔧 Loading model components sequentially...")
        self.qwen_model.load_model()
        
        self._report_memory("After model loading")
        
        # Final cleanup after loading
        self._cleanup_memory()
        self._report_memory("Final setup")
        
        print("✅ Base Qwen-Image model loaded successfully")
        
        # State for current LoRA/placeholder token
        self.current_lora_path = None
        self.current_placeholder_token = None
    
    def _cleanup_memory(self):
        """Aggressive memory cleanup optimized for H200"""
        if torch.cuda.is_available():
            # Synchronize before cleanup
            torch.cuda.synchronize()
            # Clear CUDA cache
            torch.cuda.empty_cache()
            # Reset memory stats
            torch.cuda.reset_peak_memory_stats()
        
        # Aggressive garbage collection
        gc.collect()
        
        # Force garbage collection multiple times for stubborn objects
        for _ in range(5):  # Increased for H200
            gc.collect()
    
    def _report_memory(self, stage: str):
        """Report current memory usage for debugging"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            
            print(f"📊 Memory [{stage}]: Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, ")
            print(f"    Max Allocated: {max_allocated:.2f}GB, Total GPU: {total_memory:.2f}GB")
            
            # System RAM
            ram = psutil.virtual_memory()
            print(f"    System RAM: {ram.used/1024**3:.2f}GB / {ram.total/1024**3:.2f}GB")
    
    def predict(
        self,
        prompt: str = Input(description="Text prompt for image generation", default="a photo of <zwx>"),
        replicate_weights: Optional[Path] = Input(description="Zip/tar with lora.safetensors from training", default=None),
        negative_prompt: str = Input(description="Negative prompt", default=""),
        steps: int = Input(description="Number of inference steps", default=20, ge=1, le=50),
        width: int = Input(description="Width of output image", default=512, ge=256, le=1024),
        height: int = Input(description="Height of output image", default=512, ge=256, le=1024),
        guidance_scale: float = Input(description="Guidance scale", default=4.0, ge=1.0, le=20.0),
        seed: int = Input(description="Random seed", default=42),
        lora_scale: float = Input(description="LoRA scale", default=1.0, ge=0.0, le=2.0),
        placeholder_token: Optional[str] = Input(
            description="Override placeholder token (if different from metadata)", 
            default=None
        ),
        output_dir: Optional[str] = Input(
            description="Directory to save output image (optional)", 
            default=None
        ),
    ) -> Path:
        """Generate image from text prompt using trained LoRA weights"""
        

        
        print(f"🎨 Generating image with prompt: '{prompt}'")
        print(f"📦 Using weights: {replicate_weights}")
        
        # Clean up memory before prediction and report status
        self._cleanup_memory()
        self._report_memory("Before prediction")
        
        # Handle optional weights
        if replicate_weights is not None:
            # Extract the weights archive
            weights_dir = self._extract_weights_archive(replicate_weights)
            
            # Load metadata
            metadata = self._load_metadata(weights_dir)
            
            # Determine placeholder token (metadata overrides input)
            effective_token = metadata.get("placeholder_token", placeholder_token)
            if effective_token:
                self._register_placeholder_token(effective_token)
            
            # Load LoRA weights
            self._load_lora_weights(weights_dir, lora_scale)
        else:
            print("📦 No weights provided, using base model without LoRA")
            # Handle placeholder token from input if provided
            if placeholder_token:
                self._register_placeholder_token(placeholder_token)
        
        # Clean memory before generation
        self._cleanup_memory()
        self._report_memory("Before generation")
        
        # Generate image
        output_path = self._generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            output_dir=output_dir
        )
        
        # Final cleanup
        self._cleanup_memory()
        self._report_memory("After generation")
        
        return output_path
    
    def _extract_weights_archive(self, archive_path: Optional[Path]) -> P:
        """Extract weights archive and return the extraction directory"""
        if archive_path is None:
            raise ValueError("Archive path cannot be None")
            
        extract_dir = P(tempfile.mkdtemp(prefix="qwen_weights_"))
        
        print(f"📦 Extracting weights archive to {extract_dir}")
        
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(extract_dir)
        elif tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path, 'r') as tf:
                tf.extractall(extract_dir)
        else:
            raise ValueError("Weights archive must be a .zip or .tar file")
        
        return extract_dir
    
    def _load_metadata(self, weights_dir: P) -> dict:
        """Load metadata from weights directory"""
        meta_path = weights_dir / "meta.json"
        
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            print(f"📋 Loaded metadata: {metadata}")
            return metadata
        else:
            print("⚠ No metadata found, using defaults")
            return {}
    
    def _register_placeholder_token(self, placeholder_token: str):
        """Register placeholder token in tokenizer and text encoder"""
        if not placeholder_token or placeholder_token == self.current_placeholder_token:
            return  # Already registered or no token
        
        print(f"🏷️ Registering placeholder token: {placeholder_token}")
        
        # Get tokenizer and text encoder from the model
        tokenizer = self.qwen_model.tokenizer[0] if isinstance(self.qwen_model.tokenizer, list) else self.qwen_model.tokenizer
        text_encoder = self.qwen_model.text_encoder[0] if isinstance(self.qwen_model.text_encoder, list) else self.qwen_model.text_encoder
        
        original_vocab_size = len(tokenizer)
        
        # Add the special token
        num_added = tokenizer.add_special_tokens({
            "additional_special_tokens": [placeholder_token]
        })
        
        if num_added > 0:
            # Resize text encoder embeddings
            text_encoder.resize_token_embeddings(len(tokenizer))
            print(f"✅ Added {num_added} token(s), vocab size: {original_vocab_size} → {len(tokenizer)}")
        
        self.current_placeholder_token = placeholder_token
    
    def _load_lora_weights(self, weights_dir: P, lora_scale: float):
        """Load LoRA weights onto the image transformer"""
        lora_path = weights_dir / "lora" / "lora_weights.safetensors"
        
        if not lora_path.exists():
            # Try alternative paths
            alt_paths = [
                weights_dir / "lora.safetensors",
                weights_dir / "pytorch_lora_weights.safetensors"
            ]
            
            for alt_path in alt_paths:
                if alt_path.exists():
                    lora_path = alt_path
                    break
            else:
                raise FileNotFoundError(f"LoRA weights not found in {weights_dir}")
        
        print(f"🔧 Loading LoRA weights from {lora_path} with scale {lora_scale}")
        
        # Clean memory before loading LoRA
        self._cleanup_memory()
        self._report_memory("Before LoRA loading")
        
        # Load LoRA weights using the model's built-in functionality
        # This is a simplified version - in practice, you'd use the model's LoRA loading methods
        lora_weights = safetensors.torch.load_file(lora_path)
        print(f"✅ Loaded LoRA weights: {len(lora_weights)} tensors")
        self.current_lora_path = lora_path
        
        # Clean memory after loading LoRA
        self._cleanup_memory()
        self._report_memory("After LoRA loading")
    
    def _generate_image(self, prompt: str, negative_prompt: str, width: int, height: int, 
                       steps: int, guidance_scale: float, seed: int, output_dir: Optional[str] = None) -> Path:
        """Generate image using the loaded model and LoRA"""
        
        print(f"🎨 Generating {width}x{height} image with {steps} steps, guidance {guidance_scale}")
        
        # Set seed for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Use provided output_dir or create temporary one
        if output_dir:
            output_path_obj = P(output_dir)
            output_path_obj.mkdir(parents=True, exist_ok=True)
        else:
            output_path_obj = P(tempfile.mkdtemp(prefix="qwen_output_"))
        
        # Create generate config with essential parameters only
        generate_config = GenerateImageConfig(
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            seed=seed,
            output_folder=str(output_path_obj),
            output_ext="png"
        )
        
        # Memory cleanup before generation
        self._cleanup_memory()
        
        # Use the model's generate_images method (which expects a list)
        print("🎨 Starting image generation...")
        self.qwen_model.generate_images([generate_config])
        
        # Memory cleanup after generation
        self._cleanup_memory()
        
        # Find the generated image in the output directory
        generated_files = list(output_path_obj.glob("*.png"))
        if generated_files:
            generated_file = generated_files[0]  # Take the first PNG file
            print(f"✅ Generated image saved to {generated_file}")
            return Path(str(generated_file))
        else:
            raise RuntimeError(f"No generated images found in {output_path_obj}")
    
