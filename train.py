#!/usr/bin/env python3
"""Qwen Image LoRA trainer optimized for H200 GPU"""

import os
import sys
import shutil
import zipfile
import yaml
import subprocess
import time
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# H200 optimizations
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

sys.path.insert(0, "./ai-toolkit")
from cog import BaseModel, Input, Path as CogPath

# Import safetensor conversion utility
from safetensor_utils import rename_lora_keys_for_pruna, RenameError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INPUT_DIR = Path("input_images")
OUTPUT_DIR = Path("output")
AI_TOOLKIT_PATH = Path("./ai-toolkit")


class TrainingOutput(BaseModel):
    weights: CogPath

def clean_up():
    if INPUT_DIR.exists():
        shutil.rmtree(INPUT_DIR)
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)


def extract_dataset(dataset_zip: CogPath, input_dir: Path, default_caption: str) -> Dict[str, Any]:
    input_dir.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            if not file_info.filename.startswith(("__MACOSX/", "._")):
                zip_ref.extract(file_info, input_dir)
    
    # Move files up if extracted to subdirectory
    extracted_items = list(input_dir.iterdir())
    if len(extracted_items) == 1 and extracted_items[0].is_dir():
        subdir = extracted_items[0]
        for item in subdir.iterdir():
            shutil.move(str(item), str(input_dir / item.name))
        subdir.rmdir()
    
    # Get all image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.webp']:
        image_files.extend(input_dir.glob(f"*{ext}"))
        image_files.extend(input_dir.glob(f"*{ext.upper()}"))
    
    # Create missing caption files
    created_captions = 0
    for image_file in image_files:
        caption_file = image_file.with_suffix('.txt')
        if not caption_file.exists():
            caption_file.write_text(default_caption)
            created_captions += 1
    
    total_images = len(image_files)
    existing_captions = total_images - created_captions
    
    logger.info(f"Extracted {total_images} images, {existing_captions} existing captions, {created_captions} created")
    
    return {
        "total_images": total_images,
        "total_captions": total_images,  # Now all images have captions
        "existing_captions": existing_captions,
        "created_captions": created_captions
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
                "network": {"type": "lora", "linear": lora_rank, "linear_alpha": lora_rank},
                "save": {"dtype": "float16", "save_every": steps, "max_step_saves_to_keep": 1, "push_to_hub": False},
                "datasets": [{
                    "folder_path": f"/src/{INPUT_DIR}", "default_caption": default_caption, "caption_ext": "txt",
                    "caption_dropout_rate": 0.0, "shuffle_tokens": False, "cache_latents_to_disk": False,
                    "resolution": [512, 768, 1024]
                }],
                "train": {
                    "batch_size": batch_size, "gradient_accumulation_steps": 1, "steps": steps,
                    "lr": learning_rate, "optimizer": optimizer, "lr_scheduler": "cosine",
                    "lr_scheduler_params": {"alpha": 0.75, "warmup": 0.01, "steps": steps},
                    "tf32": True, "skip_epoch": 0, "dtype": "bf16",
                    "max_grad_norm": 1.0, "gradient_checkpointing": True
                },
                "model": {
                    "name_or_path": "Qwen/Qwen2-VL-2B-Instruct", "quantize": False,
                    "transformer_layers_to_train": [24, 25, 26, 27]
                },
                "seed": seed if seed is not None else None,
                "sample": {"sampler": "rectified_flow", "sample_every": 25000, "width": 1024, "height": 1024, "seed": 42}
            }]
        }
    }


def run_training(config: Dict[str, Any], job_name: str):
    config_path = f"/tmp/{job_name}_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    cmd = ["python", str(AI_TOOLKIT_PATH / "run.py"), config_path]
    subprocess.run(cmd, check=True)


def create_output_archive(job_name: str, settings: Dict[str, Any]) -> CogPath:
    job_dir = OUTPUT_DIR / job_name
    lora_file = next(job_dir.glob("*.safetensors"))
    
    # Rename to standard name
    standard_lora_path = job_dir / "lora.safetensors"
    if lora_file != standard_lora_path:
        lora_file.rename(standard_lora_path)
    
    # Apply safetensor conversion for Pruna compatibility
    logger.info("Converting LoRA keys for Pruna compatibility...")
    try:
        conversion_result = rename_lora_keys_for_pruna(
            src_path=str(standard_lora_path),
            out_path=None,  # Overwrite in place
            dry_run=False
        )
        logger.info(f"Conversion complete: {conversion_result['num_renamed']} keys renamed")
    except RenameError as e:
        logger.warning(f"LoRA key conversion failed: {e}")
        # Continue anyway - the file might already be in the correct format
    except Exception as e:
        logger.warning(f"Unexpected error during LoRA conversion: {e}")
        # Continue anyway - better to return the file than fail completely
    
    # Create settings file
    settings_path = job_dir / "settings.txt"
    with open(settings_path, 'w') as f:
        f.write("Qwen Image LoRA Training Settings\n" + "=" * 35 + "\n\n")
        for key, value in settings.items():
            f.write(f"{key}: {value}\n")
    
    # Clean up unnecessary files
    for pattern in ["optimizer.pt", "*.safetensors"]:
        for f in job_dir.glob(pattern):
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
    
    size_mb = Path(output_path).stat().st_size / 1024 / 1024
    logger.info(f"Created {output_path} ({size_mb:.1f}MB)")
    return CogPath(output_path)


def train(
    dataset: CogPath = Input(description="ZIP file with training images and optional .txt captions"),
    steps: int = Input(default=1000, description="Training steps", ge=100, le=6000),
    learning_rate: float = Input(default=2e-4, description="Learning rate", ge=1e-5, le=1e-3),
    lora_rank: int = Input(default=64, description="LoRA rank", ge=8, le=128),
    default_caption: str = Input(default="A photo of a person named <>", description="Caption for images without matching .txt files. CRITICAL: Qwen is extremely sensitive to prompting and differs from other image models. Do NOT use abstract tokens like 'TOK', 'sks', or meaningless identifiers. Instead, use descriptive, familiar words that closely match your actual images (e.g., 'person', 'man', 'woman', 'dog', 'cat', 'building', 'car'). Every token carries meaning - the model learns by overriding specific descriptive concepts rather than learning new tokens. Be precise and descriptive about what's actually in your images. The model excels at composition and will follow detailed instructions exactly."),
    batch_size: int = Input(default=1, description="Batch size", choices=[1, 2, 4]),
    optimizer: str = Input(default="adamw", description="Optimizer", choices=["adamw8bit", "adamw", "adam8bit", "prodigy"]),
    seed: Optional[int] = Input(default=None, description="Random seed for reproducible results (leave blank for random)")
) -> TrainingOutput:
    """Train LoRA for Qwen Image - returns ZIP with lora.safetensors"""
    
    clean_up()
    job_name = f"qwen_lora_{int(time.time())}"
    
    logger.info(f"Starting training: {job_name}")
    dataset_stats = extract_dataset(dataset, INPUT_DIR, default_caption)
    
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
        "existing_captions": dataset_stats["existing_captions"],
        "created_captions": dataset_stats["created_captions"]
    }
    
    logger.info(f"Training: {steps} steps, rank {lora_rank}, {optimizer}, batch {batch_size}")
    
    run_training(config, job_name)
    output_path = create_output_archive(job_name, settings)
    
    if INPUT_DIR.exists():
        shutil.rmtree(INPUT_DIR)
    
    return TrainingOutput(weights=output_path)
