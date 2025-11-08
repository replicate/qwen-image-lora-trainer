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
from typing import Optional, Dict, Any, List
import logging
from dataclasses import dataclass

import torch

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

DEFAULT_RESOLUTIONS_HIGH_VRAM: List[int] = [512, 768, 1024]
DEFAULT_RESOLUTIONS_STANDARD: List[int] = [512, 768]
HIGH_VRAM_THRESHOLD_GB = 120.0


@dataclass(frozen=True)
class HardwareProfile:
    """Represents the GPU characteristics that matter for training."""
    device: str
    total_vram_gb: float
    name: str


def detect_hardware_profile() -> HardwareProfile:
    """Return a HardwareProfile describing the currently visible GPU."""
    if not torch.cuda.is_available():
        return HardwareProfile(device="cpu", total_vram_gb=0.0, name="cpu")
    
    device_index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)
    total_gb = props.total_memory / (1024 ** 3)
    name = props.name.strip()
    device = f"cuda:{device_index}"
    return HardwareProfile(device=device, total_vram_gb=total_gb, name=name)


class TrainingOutput(BaseModel):
    weights: CogPath

def clean_up():
    """Remove residual input/output directories from previous runs."""
    if INPUT_DIR.exists():
        shutil.rmtree(INPUT_DIR)
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)


def extract_dataset(dataset_zip: CogPath, input_dir: Path, default_caption: str) -> Dict[str, Any]:
    """Unpack the training ZIP, normalize its layout, and ensure every image has a caption."""
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
    if total_images == 0:
        raise ValueError("Dataset ZIP does not contain any supported images (.jpg/.jpeg/.png/.webp)")
    
    logger.info(f"Extracted {total_images} images, {existing_captions} existing captions, {created_captions} created")
    
    return {
        "total_images": total_images,
        "total_captions": total_images,  # Now all images have captions
        "existing_captions": existing_captions,
        "created_captions": created_captions
    }


def create_training_config(
    job_name: str,
    steps: int,
    learning_rate: float,
    lora_rank: int,
    default_caption: str,
    batch_size: int,
    optimizer: str,
    seed: Optional[int],
    hardware: HardwareProfile,
) -> Dict[str, Any]:
    """
    Build the ai-toolkit config for a single sd_trainer job.

    The config dynamically adapts to the detected GPU so the same code path
    works on 80GB A100s and 141GB H200s without manual tweaks.
    """
    is_high_vram = hardware.total_vram_gb >= HIGH_VRAM_THRESHOLD_GB
    resolutions = DEFAULT_RESOLUTIONS_HIGH_VRAM if is_high_vram else DEFAULT_RESOLUTIONS_STANDARD
    gradient_checkpointing = not is_high_vram

    return {
        "job": "extension",
        "config": {
            "name": job_name,
            "process": [{
                "type": "sd_trainer",
                "training_folder": f"/src/{OUTPUT_DIR}",
                "device": hardware.device,
                "network": {"type": "lora", "linear": lora_rank, "linear_alpha": lora_rank},
                "save": {"dtype": "float16", "save_every": steps, "max_step_saves_to_keep": 1, "push_to_hub": False},
                "datasets": [{
                    "folder_path": f"/src/{INPUT_DIR}", "default_caption": default_caption, "caption_ext": "txt",
                    "caption_dropout_rate": 0.0, "shuffle_tokens": False, "cache_latents_to_disk": False,
                    "resolution": resolutions, "pin_memory": True, "num_workers": 4
                }],
                "train": {
                    "batch_size": batch_size, "steps": steps, "gradient_accumulation_steps": 2,
                    "train_unet": True, "train_text_encoder": False, "gradient_checkpointing": gradient_checkpointing,
                    "noise_scheduler": "flowmatch", "optimizer": optimizer, "lr": learning_rate,
                    "dtype": "bf16", "max_grad_norm": 1.0, "seed": seed or 42,
                    "ema_config": {"use_ema": False, "ema_decay": 0.99}
                },
                "model": {"name_or_path": "Qwen/Qwen-Image", "arch": "qwen_image", "quantize": False, "quantize_te": False, "low_vram": False},
                "sample": {"sample_every": 0}
            }],
            "meta": {"name": "[name]", "version": "1.0"}
        }
    }


def run_training(config: Dict[str, Any], job_name: str) -> None:
    """Write the ai-toolkit config to disk and run the training subprocess."""
    job_dir = OUTPUT_DIR / job_name
    job_dir.mkdir(parents=True, exist_ok=True)
    config_path = job_dir / "config.yaml"
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    env = os.environ.copy()
    env.update({
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:512",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_ASYNC_ERROR_HANDLING": "1"
    })
    
    logger.info(f"Starting training: {job_name}")
    subprocess.run([sys.executable, "run.py", str(config_path.absolute())], 
                   cwd=str(AI_TOOLKIT_PATH), env=env, check=True)
    logger.info("Training completed")


def create_output_archive(job_name: str, settings: Dict[str, Any]) -> CogPath:
    """Package the trained LoRA weights and metadata into a ZIP file for download."""
    job_dir = OUTPUT_DIR / job_name
    lora_candidates = sorted(job_dir.glob("*.safetensors"))
    if not lora_candidates:
        raise FileNotFoundError(f"No .safetensors found in {job_dir}")
    lora_file = lora_candidates[0]
    
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
    learning_rate: float = Input(default=5e-4, description="Learning rate", ge=1e-5, le=1e-3),
    lora_rank: int = Input(default=32, description="LoRA rank", ge=8, le=128),
    default_caption: str = Input(default="A photo of a person named <>", description="Caption for images without matching .txt files. CRITICAL: Qwen is extremely sensitive to prompting and differs from other image models. Do NOT use abstract tokens like 'TOK', 'sks', or meaningless identifiers. Instead, use descriptive, familiar words that closely match your actual images (e.g., 'person', 'man', 'woman', 'dog', 'cat', 'building', 'car'). Every token carries meaning - the model learns by overriding specific descriptive concepts rather than learning new tokens. Be precise and descriptive about what's actually in your images. The model excels at composition and will follow detailed instructions exactly."),
    batch_size: int = Input(default=1, description="Batch size", choices=[1, 2, 4]),
    optimizer: str = Input(default="adamw", description="Optimizer", choices=["adamw8bit", "adamw", "adam8bit", "prodigy"]),
    seed: Optional[int] = Input(default=None, description="Random seed for reproducible results (leave blank for random)")
) -> TrainingOutput:
    """Train LoRA for Qwen Image - returns ZIP with lora.safetensors"""
    
    clean_up()
    job_name = f"qwen_lora_{int(time.time())}"
    hardware = detect_hardware_profile()
    logger.info(
        "Detected GPU: %s (%.1f GB) on %s",
        hardware.name,
        hardware.total_vram_gb,
        hardware.device,
    )
    
    logger.info(f"Starting training: {job_name}")
    dataset_stats = extract_dataset(dataset, INPUT_DIR, default_caption)
    
    config = create_training_config(
        job_name,
        steps,
        learning_rate,
        lora_rank,
        default_caption,
        batch_size,
        optimizer,
        seed,
        hardware,
    )
    process_config = config["config"]["process"][0]
    train_config = process_config["train"]
    dataset_config = process_config["datasets"][0]
    logger.info(
        "Resolved training profile: resolutions=%s, gradient_checkpointing=%s",
        dataset_config["resolution"],
        train_config["gradient_checkpointing"],
    )

    # Training settings for output
    settings = {
        "steps": steps,
        "learning_rate": learning_rate,
        "lora_rank": lora_rank,
        "lora_alpha": lora_rank,
        "batch_size": batch_size,
        "optimizer": optimizer,
        "seed": seed if seed is not None else "random",
        "resolution": str(dataset_config["resolution"]),
        "default_caption": default_caption,
        "images": dataset_stats["total_images"],
        "existing_captions": dataset_stats["existing_captions"],
        "created_captions": dataset_stats["created_captions"],
        "gradient_checkpointing": train_config["gradient_checkpointing"],
        "device": hardware.device,
        "gpu_name": hardware.name,
        "gpu_total_vram_gb": round(hardware.total_vram_gb, 1),
    }
    
    logger.info(f"Training: {steps} steps, rank {lora_rank}, {optimizer}, batch {batch_size}")
    
    run_training(config, job_name)
    output_path = create_output_archive(job_name, settings)
    
    if INPUT_DIR.exists():
        shutil.rmtree(INPUT_DIR)
    
    return TrainingOutput(weights=output_path)
