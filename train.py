#!/usr/bin/env python3
"""Qwen Image LoRA trainer optimized for H200 GPU"""

import os
import sys
import tempfile
import shutil
import zipfile
import yaml
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

# H200 optimizations
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

sys.path.insert(0, "./ai-toolkit")
from cog import Input, Path as CogPath

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INPUT_DIR = Path("input_images")
OUTPUT_DIR = Path("output")
AI_TOOLKIT_PATH = Path("./ai-toolkit")

def clean_up():
    if INPUT_DIR.exists():
        shutil.rmtree(INPUT_DIR)
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)


def extract_dataset(dataset_zip: CogPath, input_dir: Path) -> Dict[str, Any]:
    
    input_dir.mkdir(parents=True, exist_ok=True)
    
    image_count = 0
    caption_count = 0
    
    with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            if not file_info.filename.startswith(("__MACOSX/", "._")):
                zip_ref.extract(file_info, input_dir)
                if file_info.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    image_count += 1
                elif file_info.filename.lower().endswith('.txt'):
                    caption_count += 1
    
    # Move files up if extracted to subdirectory
    extracted_items = list(input_dir.iterdir())
    if len(extracted_items) == 1 and extracted_items[0].is_dir():
        subdir = extracted_items[0]
        for item in subdir.iterdir():
            shutil.move(str(item), str(input_dir / item.name))
        subdir.rmdir()
        
        # Recount
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.webp']:
            image_files.extend(input_dir.glob(f"*{ext}"))
            image_files.extend(input_dir.glob(f"*{ext.upper()}"))
        image_count = len(image_files)
        caption_count = len(list(input_dir.glob("*.txt")))
    
    logger.info(f"Extracted {image_count} images and {caption_count} caption files")
    
    return {
        "total_images": image_count,
        "total_captions": caption_count,
        "caption_coverage": caption_count / image_count if image_count > 0 else 0
    }


def create_training_config(job_name: str, steps: int, learning_rate: float, lora_rank: int, 
                          default_caption: str, batch_size: int, optimizer: str, seed: Optional[int]) -> Dict[str, Any]:
    return {
        "job": "extension",
        "config": {
            "name": job_name,
            "process": [{
                "type": "sd_trainer",
                "training_folder": f"/src/{OUTPUT_DIR}",
                "device": "cuda:0",
                "network": {
                    "type": "lora",
                    "linear": lora_rank,
                    "linear_alpha": lora_rank
                },
                "save": {
                    "dtype": "float16",
                    "save_every": steps,
                    "max_step_saves_to_keep": 1,
                    "push_to_hub": False
                },
                "datasets": [{
                    "folder_path": f"/src/{INPUT_DIR}",
                    "default_caption": default_caption,
                    "caption_ext": "txt",
                    "caption_dropout_rate": 0.0,
                    "shuffle_tokens": False,
                    "cache_latents_to_disk": False,
                    "resolution": [512, 768, 1024],
                    "pin_memory": True,
                    "num_workers": 4
                }],
                "train": {
                    "batch_size": batch_size,
                    "steps": steps,
                    "gradient_accumulation_steps": 2,
                    "train_unet": True,
                    "train_text_encoder": False,
                    "gradient_checkpointing": False,
                    "noise_scheduler": "flowmatch",
                    "optimizer": optimizer,
                    "lr": learning_rate,
                    "dtype": "bf16",
                    "max_grad_norm": 1.0,
                    "seed": seed if seed is not None else 42,
                    "ema_config": {
                        "use_ema": False,
                        "ema_decay": 0.99
                    }
                },
                "model": {
                    "name_or_path": "Qwen/Qwen-Image",
                    "arch": "qwen_image",
                    "quantize": False,
                    "quantize_te": False,
                    "low_vram": False
                },
                "sample": {
                    "sample_every": 0
                }
            }],
            "meta": {
                "name": "[name]",
                "version": "1.0"
            }
        }
    }


def run_training(config: Dict[str, Any], job_name: str) -> None:
    job_dir = OUTPUT_DIR / job_name
    job_dir.mkdir(parents=True, exist_ok=True)
    config_path = job_dir / "config.yaml"
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
    env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    env["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    
    cmd = [sys.executable, "run.py", str(config_path.absolute())]
    logger.info(f"Starting training: {job_name}")
    
    subprocess.run(cmd, cwd=str(AI_TOOLKIT_PATH), env=env, check=True)
    logger.info("Training completed")


def create_output_archive(job_name: str, settings: Dict[str, Any]) -> CogPath:
    job_dir = OUTPUT_DIR / job_name
    safetensors_files = list(job_dir.glob("*.safetensors"))
    lora_file = safetensors_files[0]
    
    # Rename to standard name
    standard_lora_path = job_dir / "lora.safetensors"
    if lora_file != standard_lora_path:
        lora_file.rename(standard_lora_path)
    
    # Create simple settings file
    settings_path = job_dir / "settings.txt"
    with open(settings_path, 'w') as f:
        f.write("Qwen Image LoRA Training Settings\n")
        f.write("=" * 35 + "\n\n")
        for key, value in settings.items():
            f.write(f"{key}: {value}\n")
    
    # Clean up
    for f in job_dir.glob("optimizer.pt"):
        f.unlink()
    for f in job_dir.glob("*.safetensors"):
        if f.name != "lora.safetensors":
            f.unlink()
    
    # Create ZIP
    output_path = f"/tmp/{job_name}_trained.zip"
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(standard_lora_path, "lora.safetensors")
        zipf.write(settings_path, "settings.txt")
        
        config_file = job_dir / "config.yaml"
        if config_file.exists():
            zipf.write(config_file, "config.yaml")
    
    logger.info(f"Created {output_path} ({Path(output_path).stat().st_size / 1024 / 1024:.1f}MB)")
    return CogPath(output_path)


def train(
    dataset: CogPath = Input(description="ZIP file with training images and optional .txt captions"),
    steps: int = Input(default=1000, description="Training steps", ge=100, le=6000),
    learning_rate: float = Input(default=1e-4, description="Learning rate", ge=1e-5, le=1e-3),
    lora_rank: int = Input(default=32, description="LoRA rank", ge=8, le=128),
    default_caption: str = Input(default="A photo of a man named zeke", description="Default caption for images without .txt files"),
    batch_size: int = Input(default=1, description="Batch size", choices=[1, 2, 4]),
    optimizer: str = Input(default="adamw", description="Optimizer", choices=["adamw8bit", "adamw", "adam8bit", "prodigy"]),
    seed: Optional[int] = Input(default=None, description="Random seed for reproducible results (leave blank for random)")
) -> CogPath:
    """Train LoRA for Qwen Image - returns ZIP with lora.safetensors"""
    
    clean_up()
    job_name = f"qwen_lora_{int(time.time())}"
    
    logger.info(f"Starting training: {job_name}")
    dataset_stats = extract_dataset(dataset, INPUT_DIR)
    
    config = create_training_config(job_name, steps, learning_rate, lora_rank, 
                                   default_caption, batch_size, optimizer, seed)
    
    # Training settings for output
    settings = {
        "steps": steps,
        "learning_rate": learning_rate,
        "lora_rank": lora_rank,
        "lora_alpha": lora_rank,
        "batch_size": batch_size,
        "optimizer": optimizer,
        "seed": seed if seed is not None else "random",
        "resolution": "[512, 768, 1024]",
        "default_caption": default_caption,
        "images": dataset_stats["total_images"],
        "captions": dataset_stats["total_captions"]
    }
    
    logger.info(f"Training: {steps} steps, rank {lora_rank}, {optimizer}, batch {batch_size}")
    
    run_training(config, job_name)
    output_path = create_output_archive(job_name, settings)
    
    if INPUT_DIR.exists():
        shutil.rmtree(INPUT_DIR)
    
    return output_path