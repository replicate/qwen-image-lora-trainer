#!/usr/bin/env python3
"""
Cog Training Interface for Qwen-Image LoRA with Zero-Prior Placeholder Tokens

This module provides the Cog training API that:
1. Accepts training inputs (dataset, configuration)
2. Runs the ai-toolkit training pipeline
3. Packages LoRA weights + metadata into a single archive
4. Returns the archive for use in prediction
"""

import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path as P
from typing import Optional

import yaml
from cog import BaseModel, Input, Path
from pydantic import Field


class TrainingOutput(BaseModel):
    """Output from Cog training - contains the weights file for prediction"""
    weights: Path = Field(description="Archive containing LoRA weights and metadata")


def train(
    dataset: Path = Input(description="Zip file containing training images and captions"),
    train_placeholder_token: bool = Input(
        default=False,
        description="Enable zero-prior placeholder token training (default OFF)"
    ),
    placeholder_token: str = Input(
        default="<zwx>", 
        description="Zero-prior placeholder token (use angle brackets for single-token encoding)"
    ),
    rank: int = Input(default=64, ge=4, le=256, description="LoRA rank"),
    steps: int = Input(default=4000, ge=100, le=10000, description="Training steps"),
    learning_rate: float = Input(default=1e-4, ge=1e-6, le=1e-2, description="Learning rate"),
    batch_size: int = Input(default=1, ge=1, le=8, description="Batch size"),
    save_every: int = Input(default=2000, ge=100, description="Save checkpoint every N steps"),
    resolution: str = Input(
        default="512,768,1024", 
        description="Training resolutions (comma-separated)"
    ),
    placeholder_init_std: float = Input(
        default=1e-5, 
        ge=0.0, 
        le=0.1, 
        description="Placeholder token initialization standard deviation"
    ),
    enable_debug_logging: bool = Input(
        default=False, 
        description="Enable debug logging for prompt tokenization"
    )
) -> TrainingOutput:
    """
    Train a Qwen-Image LoRA with optional zero-prior placeholder token
    
    Returns a weights archive that can be used with cog predict
    """
    print("🚀 Starting Qwen-Image LoRA training with zero-prior placeholder token")
    
    # Create working directory
    work_dir = P(tempfile.mkdtemp(prefix="qwen_train_"))
    print(f"Working directory: {work_dir}")
    
    try:
        # 1. Extract and prepare dataset
        dataset_dir = work_dir / "dataset"
        dataset_dir.mkdir()
        
        print("📦 Extracting training dataset...")
        with zipfile.ZipFile(dataset, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
        
        # Validate dataset structure
        image_files = list(dataset_dir.glob("*.jpg")) + list(dataset_dir.glob("*.png")) + list(dataset_dir.glob("*.jpeg"))
        if not image_files:
            raise ValueError("No image files found in dataset zip")
        
        caption_files = list(dataset_dir.glob("*.txt"))
        print(f"Found {len(image_files)} images and {len(caption_files)} captions")
        
        # 2. Generate training configuration
        config_path = work_dir / "train_config.yaml"
        
        # Parse resolution list
        resolutions = [int(r.strip()) for r in resolution.split(',')]
        
        config = {
            "job": "extension",
            "config": {
                "name": f"qwen_lora_cog_{placeholder_token.replace('<', '').replace('>', '')}",
                "process": [{
                    "type": "sd_trainer",
                    "training_folder": str(work_dir / "output"),
                    "device": "cuda:0",
                    "network": {
                        "type": "lora",
                        "linear": rank,
                        "linear_alpha": rank * 2
                    },
                    "save": {
                        "dtype": "float16",
                        "save_every": save_every,
                        "max_step_saves_to_keep": 2,
                        "push_to_hub": False
                    },
                    "datasets": [{
                        "folder_path": str(dataset_dir),
                        "caption_ext": "txt",
                        "caption_dropout_rate": 0.0,
                        "shuffle_tokens": False,
                        "cache_latents_to_disk": True,
                        "resolution": resolutions
                    }],
                    "train": {
                        "batch_size": batch_size,
                        "steps": steps,
                        "gradient_accumulation_steps": 1,
                        "train_unet": True,
                        "train_text_encoder": False,
                        "gradient_checkpointing": True,
                        "noise_scheduler": "flowmatch",
                        "optimizer": "adamw8bit",
                        "lr": learning_rate,
                        "dtype": "bf16",
                        "ema_config": {
                            "use_ema": True,
                            "ema_decay": 0.99
                        },
                        # Zero-prior placeholder token settings
                        "train_placeholder_token": train_placeholder_token,
                        "placeholder_token": placeholder_token,
                        "placeholder_init_std": placeholder_init_std
                    },
                    "debug": {
                        "log_prompt_ids": enable_debug_logging
                    },
                    "model": {
                        "name_or_path": "Qwen/Qwen-Image",
                        "arch": "qwen_image",
                        "quantize": True,
                        "quantize_te": True,
                        "low_vram": True
                    },
                    "sample": {
                        "sampler": "flowmatch",
                        "sample_every": save_every // 4,  # Sample more frequently than saves
                        "width": 1024,
                        "height": 1024,
                        "prompts": [
                            f"a photo of {placeholder_token}",
                            f"a portrait of {placeholder_token} smiling",
                            f"{placeholder_token} in a garden",
                            f"beautiful {placeholder_token} at sunset"
                        ],
                        "seed": 42,
                        "walk_seed": True,
                        "guidance_scale": 4.0,
                        "sample_steps": 20
                    }
                }],
                "meta": {
                    "name": "[name]",
                    "version": "1.0"
                }
            }
        }
        
        # Write configuration
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print("📝 Generated training configuration")
        if enable_debug_logging:
            print(f"Config: {config_path}")
        
        # 3. Run training
        print("🎯 Starting LoRA training...")
        
        # Change to ai-toolkit directory and run training
        ai_toolkit_dir = P(__file__).parent / "ai-toolkit"
        if not ai_toolkit_dir.exists():
            raise RuntimeError(
                "ai-toolkit directory not found. Make sure submodule is initialized with:\n"
                "git submodule update --init --recursive"
            )
        
        # Verify ai-toolkit run.py exists
        run_py_path = ai_toolkit_dir / "run.py"
        if not run_py_path.exists():
            raise RuntimeError(
                f"ai-toolkit/run.py not found at {run_py_path}. "
                "Check ai-toolkit submodule initialization."
            )
        
        # Run the training command
        cmd = [
            "python", "run.py", str(config_path)
        ]
        
        print(f"Running: {' '.join(cmd)}")
        print(f"Working directory: {ai_toolkit_dir}")
        
        result = subprocess.run(
            cmd,
            cwd=ai_toolkit_dir,
            capture_output=True,
            text=True,
            check=True
        )
        
        print("✅ Training completed successfully")
        if enable_debug_logging:
            print("Training output:")
            print(result.stdout)
        
        # 4. Package results
        output_dir = work_dir / "output"
        
        # Find the latest LoRA weights
        lora_files = list(output_dir.glob("**/lora_weights.safetensors"))
        if not lora_files:
            # Try alternative paths
            lora_files = list(output_dir.glob("**/*.safetensors"))
        
        if not lora_files:
            raise RuntimeError("No LoRA weights found after training")
        
        # Use the most recent LoRA file
        lora_path = max(lora_files, key=os.path.getmtime)
        print(f"📦 Found LoRA weights: {lora_path}")
        
        # Create the weights archive
        weights_dir = work_dir / "weights_package"
        weights_dir.mkdir()
        lora_dir = weights_dir / "lora"
        lora_dir.mkdir()
        
        # Copy LoRA weights
        shutil.copy2(lora_path, lora_dir / "lora_weights.safetensors")
        
        # Create metadata
        metadata = {
            "base_model": "Qwen/Qwen-Image",
            "train_placeholder_token": train_placeholder_token,
            "placeholder_token": placeholder_token if train_placeholder_token else None,
            "placeholder_init_std": placeholder_init_std if train_placeholder_token else None,
            "rank": rank,
            "training_steps": steps,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "resolutions": resolutions,
            "created_with": "qwen-image-lora-trainer",
            "version": "1.0"
        }
        
        # Try to get vocab size from tokenizer if possible
        if train_placeholder_token:
            try:
                from transformers import Qwen2Tokenizer
                tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen-Image", subfolder="tokenizer")
                tokenizer.add_special_tokens({"additional_special_tokens": [placeholder_token]})
                metadata["vocab_size"] = len(tokenizer)
                metadata["placeholder_token_id"] = tokenizer.convert_tokens_to_ids(placeholder_token)
            except Exception as e:
                print(f"⚠ Could not determine vocab size: {e}")
        
        # Write metadata
        with open(weights_dir / "meta.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create final zip archive
        weights_zip = work_dir / "weights.zip"
        with zipfile.ZipFile(weights_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(lora_dir / "lora_weights.safetensors", "lora/lora_weights.safetensors")
            zf.write(weights_dir / "meta.json", "meta.json")
        
        print(f"✅ Created weights archive: {weights_zip}")
        print(f"📊 Archive size: {weights_zip.stat().st_size / 1024 / 1024:.1f} MB")
        
        # Copy to persistent location for Cog
        final_weights = P("/tmp/weights.zip")
        shutil.copy2(weights_zip, final_weights)
        
        return TrainingOutput(weights=Path(str(final_weights)))
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise RuntimeError(f"Training process failed: {e.stderr}")
    
    except Exception as e:
        print(f"❌ Training error: {e}")
        raise
    
    finally:
        # Cleanup working directory (optional - comment out for debugging)
        if not enable_debug_logging:
            try:
                shutil.rmtree(work_dir)
            except Exception as e:
                print(f"⚠ Could not clean up {work_dir}: {e}")


if __name__ == "__main__":
    # For local testing
    import tempfile
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python train.py <dataset.zip>")
        sys.exit(1)
    
    dataset_path = Path(sys.argv[1])
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        sys.exit(1)
    
    print("Running local training test...")
    result = train(dataset=dataset_path)
    print(f"Training completed. Weights: {result.weights}")