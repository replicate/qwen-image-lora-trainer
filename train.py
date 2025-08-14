#!/usr/bin/env python3
"""
Cog trainer for Qwen Image LoRA training.
Optimized for H200 GPU performance based on experimental benchmarks.

H200 Optimized Defaults (based on 5-experiment sweep):
- LoRA rank=16 (50% smaller models, 3% training overhead vs rank=32)
- Resolution=1024 (full quality on H200's ample VRAM)
- Steps=1000 (efficient training duration)
- gradient_checkpointing=False, low_vram=False (speed-first on H200)
- Optimizer=adamw (fastest convergence)

Hardware profiles:
- H200 (recommended): Use defaults - optimized for speed and efficiency
- A100 (memory-constrained): Set gradient_checkpointing=True, low_vram=True, resolution=512
"""

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

# Set environment variables for optimal performance
# A100 profile: these defaults helped stabilize allocation under tighter VRAM.
# H200 profile: speed-focused; allocator tweak is harmless but optional.
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Add ai-toolkit to path
sys.path.insert(0, "./ai-toolkit")

from cog import BaseModel, Input, Path as CogPath

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
JOB_NAME_PREFIX = "qwen_lora"
INPUT_DIR = Path("input_images")
OUTPUT_DIR = Path("output")
AI_TOOLKIT_PATH = Path("./ai-toolkit")


class TrainingOutput(BaseModel):
    """Output model for training results"""
    weights: CogPath


def clean_up():
    """Clean up any existing training directories"""
    if INPUT_DIR.exists():
        shutil.rmtree(INPUT_DIR)
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)


def extract_dataset(dataset_zip: CogPath, input_dir: Path) -> Dict[str, Any]:
    """
    Extract dataset zip file and return statistics
    
    Args:
        dataset_zip: Path to input zip file
        input_dir: Directory to extract to
        
    Returns:
        Dict with dataset statistics
    """
    if not zipfile.is_zipfile(dataset_zip):
        raise ValueError("Input must be a zip file")
    
    input_dir.mkdir(parents=True, exist_ok=True)
    
    image_count = 0
    caption_count = 0
    
    with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            # Skip macOS metadata files
            if not file_info.filename.startswith("__MACOSX/") and not file_info.filename.startswith("._"):
                zip_ref.extract(file_info, input_dir)
                
                # Count files
                if file_info.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    image_count += 1
                elif file_info.filename.lower().endswith('.txt'):
                    caption_count += 1
    
    # If extracted to a subdirectory, move files up
    extracted_items = list(input_dir.iterdir())
    if len(extracted_items) == 1 and extracted_items[0].is_dir():
        subdir = extracted_items[0]
        for item in subdir.iterdir():
            shutil.move(str(item), str(input_dir / item.name))
        subdir.rmdir()
        
        # Recount after moving
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.webp']:
            image_files.extend(input_dir.glob(f"*{ext}"))
            image_files.extend(input_dir.glob(f"*{ext.upper()}"))
        caption_files = list(input_dir.glob("*.txt"))
        
        image_count = len(image_files)
        caption_count = len(caption_files)
    
    if image_count == 0:
        raise ValueError("No images found in the dataset. Supported formats: JPG, PNG, JPEG, WebP")
    
    if image_count < 5:
        logger.warning(f"Very few images found ({image_count}). Consider adding more for better results.")
    
    logger.info(f"Extracted {image_count} images and {caption_count} caption files")
    
    return {
        "total_images": image_count,
        "total_captions": caption_count,
        "caption_coverage": caption_count / image_count if image_count > 0 else 0
    }


def create_training_config(
    job_name: str,
    dataset_stats: Dict[str, Any],
    steps: int,
    learning_rate: float,
    lora_rank: int,
    lora_alpha: int,
    resolution: int,
    default_caption: str,
    caption_dropout_rate: float,
    batch_size: int,
    optimizer: str,
    sample_every: int,
    sample_prompts: List[str],
    gradient_checkpointing: bool,
    low_vram: bool,
    use_ema: bool,
    ema_decay: float
) -> Dict[str, Any]:
    """Create ai-toolkit compatible training configuration"""
    
    # Parse sample prompts
    if not sample_prompts or (len(sample_prompts) == 1 and not sample_prompts[0].strip()):
        sample_prompts = [default_caption]
    
    # H200 optimized defaults based on experimental results:
    # - Best efficiency: rank=16, resolution=1024, gradient_checkpointing=False, low_vram=False
    # - 50% smaller model size vs rank=32, only 3% slower training
    # - A100 fallback: gradient_checkpointing=True, low_vram=True, resolution=512
    config = {
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
                    "linear_alpha": lora_alpha
                },
                "save": {
                    "dtype": "float16",
                    "save_every": steps,  # Save only at completion
                    "max_step_saves_to_keep": 1,
                    "push_to_hub": False
                },
                "datasets": [{
                    "folder_path": f"/src/{INPUT_DIR}",
                    "default_caption": default_caption,
                    "caption_ext": "txt",
                    "caption_dropout_rate": caption_dropout_rate,
                    "shuffle_tokens": False,
                    "cache_latents_to_disk": True,
                    "resolution": [resolution]  # Single resolution for memory efficiency
                }],
                "train": {
                    "batch_size": batch_size,
                    "steps": steps,
                    "gradient_accumulation_steps": 1,
                    "train_unet": True,
                    "train_text_encoder": False,
                    "gradient_checkpointing": gradient_checkpointing,
                    "noise_scheduler": "flowmatch",  # Qwen Image uses flowmatch
                    "optimizer": optimizer,
                    "lr": learning_rate,
                    "dtype": "bf16",
                    "max_grad_norm": 1.0,
                    "ema_config": {
                        "use_ema": use_ema,
                        "ema_decay": ema_decay
                    }
                },
                "model": {
                    "name_or_path": "Qwen/Qwen-Image",
                    "arch": "qwen_image",
                    "quantize": False,
                    "quantize_te": False,
                    # H200 (default): low_vram=False maximizes throughput with ample VRAM
                    # A100 fallback: set low_vram=True to avoid OOMs
                    "low_vram": low_vram
                },
                "sample": {
                    "sampler": "flowmatch",
                    "sample_every": sample_every,
                    # H200 (default): sampling at 1024 leverages full VRAM efficiently
                    # A100: reduce to 512 for memory conservation. Set sample_every=0 for max speed.
                    "width": resolution,
                    "height": resolution,
                    "prompts": sample_prompts,
                    "seed": 42,
                    "walk_seed": True,
                    "guidance_scale": 4,
                    "sample_steps": 20
                } if sample_every > 0 else {
                    "sample_every": 0  # Disable sampling
                }
            }],
            "meta": {
                "name": "[name]",
                "version": "1.0"
            }
        }
    }
    
    return config


def run_training(config: Dict[str, Any], job_name: str) -> None:
    """Execute the training process"""
    
    if not AI_TOOLKIT_PATH.exists():
        raise RuntimeError(f"ai-toolkit not found at {AI_TOOLKIT_PATH}")
    
    # Write config to per-job folder to ensure packaging picks it up
    job_dir = OUTPUT_DIR / job_name
    job_dir.mkdir(parents=True, exist_ok=True)
    config_path = job_dir / "config.yaml"
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    # Prepare environment
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Build command
    cmd = [sys.executable, "run.py", str(config_path.absolute())]
    
    logger.info(f"Starting training with command: {' '.join(cmd)}")
    logger.info(f"Working directory: {AI_TOOLKIT_PATH}")
    logger.info(f"Config file: {config_path}")
    
    # Run training
    try:
        result = subprocess.run(
            cmd,
            cwd=str(AI_TOOLKIT_PATH),
            env=env,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout
        )
        
        if result.returncode != 0:
            error_msg = f"Training failed with exit code {result.returncode}"
            if result.stderr:
                error_msg += f"\nStderr (last 2000 chars): {result.stderr[-2000:]}"
            if result.stdout:
                error_msg += f"\nStdout (last 2000 chars): {result.stdout[-2000:]}"
            raise RuntimeError(error_msg)
            
        logger.info("Training completed successfully")
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("Training timed out after 2 hours")


def create_output_archive(job_name: str, dataset_stats: Dict[str, Any]) -> CogPath:
    """Create and return the final output zip archive"""
    
    job_dir = OUTPUT_DIR / job_name
    
    if not job_dir.exists():
        raise RuntimeError(f"Training output directory not found: {job_dir}")
    
    # Find the LoRA file
    safetensors_files = list(job_dir.glob("*.safetensors"))
    if not safetensors_files:
        raise RuntimeError("No safetensors files generated by training")
    
    lora_file = safetensors_files[0]
    logger.info(f"Found trained LoRA: {lora_file.name} ({lora_file.stat().st_size / 1024 / 1024:.1f}MB)")
    
    # Create output zip archive
    output_path = f"/tmp/{job_name}_trained.zip"
    
    # Rename LoRA file to standard name
    standard_lora_path = job_dir / "lora.safetensors"
    if lora_file != standard_lora_path:
        lora_file.rename(standard_lora_path)
    
    # Create metadata file
    metadata = {
        "job_name": job_name,
        "model_name": "Qwen/Qwen-Image",
        "model_type": "qwen_image",
        "dataset_stats": dataset_stats,
        "lora_file_size_mb": standard_lora_path.stat().st_size / 1024 / 1024,
        "training_completed": True,
        "timestamp": time.time()
    }
    
    metadata_path = job_dir / "metadata.yaml"
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False, indent=2)
    
    # Clean up unwanted files
    optimizer_file = job_dir / "optimizer.pt"
    if optimizer_file.exists():
        optimizer_file.unlink()
    
    # Remove intermediate LoRA files (keep only lora.safetensors)
    for safetensors_file in job_dir.glob("*.safetensors"):
        if safetensors_file.name != "lora.safetensors":
            safetensors_file.unlink()
    
    # Create zip archive
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add main LoRA file
        zipf.write(standard_lora_path, "lora.safetensors")
        
        # Add metadata
        zipf.write(metadata_path, "metadata.yaml")
        
        # Add training config if it exists
        config_file = job_dir / "config.yaml"
        if config_file.exists():
            zipf.write(config_file, "training_config.yaml")
        
        # Add sample images if they exist
        samples_dir = job_dir / "samples"
        if samples_dir.exists():
            for sample_file in samples_dir.glob("*.jpg"):
                zipf.write(sample_file, f"samples/{sample_file.name}")
    
    if not Path(output_path).exists():
        raise RuntimeError("Failed to create output zip archive")
    
    logger.info(f"Created output zip: {output_path} ({Path(output_path).stat().st_size / 1024 / 1024:.1f}MB)")
    
    return CogPath(output_path)


def train(
    dataset: CogPath = Input(
        description="Zip file containing training images. Images should be JPG/PNG/JPEG/WebP. Optional .txt files with same names as images for captions."
    ),
    steps: int = Input(
        default=1000,
        description="Number of training steps. 1000 provides good balance for H200 efficiency. Range: 500-4000. Higher values = more training but risk overfitting.",
        ge=100,
        le=6000
    ),
    learning_rate: float = Input(
        default=1e-4,
        description="Learning rate for training. Higher = faster learning but less stable. Recommended: 1e-4 to 5e-5 for most cases.",
        ge=1e-5,
        le=1e-3
    ),
    lora_rank: int = Input(
        default=16,
        description="LoRA rank. Higher ranks can capture more complex features but take longer to train and create larger files. 16 provides optimal efficiency on H200.",
        ge=8,
        le=128
    ),
    lora_alpha: int = Input(
        default=16,
        description="LoRA alpha parameter. Usually set equal to rank for balanced training. Affects the strength of the LoRA.",
        ge=8,
        le=128
    ),
    resolution: int = Input(
        default=1024,
        description="Training resolution. Higher resolutions produce better quality but use more memory. 1024 recommended for H200.",
        choices=[512, 768, 1024]
    ),
    default_caption: str = Input(
        default="A photo of a man named zeke",
        description="Default caption used for images without .txt caption files. Use something descriptive like 'A photo of [subject_name]' or 'artwork in [style_name] style'."
    ),
    caption_dropout_rate: float = Input(
        default=0.05,
        description="Probability of dropping captions during training (0.05 = 5% of steps will ignore captions). Helps with unconditional generation and prevents overfitting to text.",
        ge=0.0,
        le=0.3
    ),
    batch_size: int = Input(
        default=1,
        description="Batch size for training. H200 can handle larger batches, but 1 provides stable convergence for LoRA training.",
        choices=[1, 2, 4]
    ),
    optimizer: str = Input(
        default="adamw",
        description="Optimizer for training. adamw is speed-optimized for H200. adamw8bit is memory-efficient for smaller GPUs.",
        choices=["adamw8bit", "adamw", "adam8bit", "prodigy"]
    ),
    sample_every: int = Input(
        default=250,
        description="Generate sample images every N steps to monitor training progress. Set to 0 to disable sampling (saves time).",
        ge=0,
        le=1000
    ),
    sample_prompts: str = Input(
        default="",
        description="Custom prompts for sampling, separated by semicolons (;). If empty, uses default_caption. Example: 'portrait of subject;subject smiling;subject in different pose'"
    ),
    gradient_checkpointing: bool = Input(
        default=False,
        description="Enable gradient checkpointing to save memory at the cost of some training speed. False recommended for H200 speed-first."
    ),
    low_vram: bool = Input(
        default=False,
        description="Enable low VRAM optimizations. False recommended for H200 with ample VRAM for maximum speed."
    ),
    use_ema: bool = Input(
        default=False,
        description="Use Exponential Moving Average for more stable training but slower speed. False recommended for H200 speed-first."
    ),
    ema_decay: float = Input(
        default=0.99,
        description="EMA decay rate. Higher values = more smoothing. 0.99 is a good default.",
        ge=0.9,
        le=0.999
    ),
    cache_latents_to_disk: bool = Input(
        default=False,
        description="Cache latents to disk to save memory. Enable this if you have a large dataset and encounter out-of-memory errors."
    )
) -> TrainingOutput:
    """
    Train a LoRA (Low-Rank Adaptation) model for Qwen Image.
    
    This will create a LoRA that can be used with the base Qwen Image model
    to generate images in a specific style or of a specific subject.
    
    Returns a zip file containing the trained LoRA weights and metadata.
    """
    
    logger.info("Starting Qwen Image LoRA training...")
    
    # Clean up any previous runs
    clean_up()
    
    # Generate unique job name
    job_name = f"{JOB_NAME_PREFIX}_{int(time.time())}"
    logger.info(f"Job name: {job_name}")
    
    try:
        # Extract and validate dataset
        logger.info("Extracting and validating dataset...")
        dataset_stats = extract_dataset(dataset, INPUT_DIR)
        
        # Parse sample prompts
        if sample_prompts.strip():
            prompts_list = [p.strip() for p in sample_prompts.split(';') if p.strip()]
        else:
            prompts_list = [default_caption]
        
        logger.info(f"Sample prompts: {prompts_list}")
        
        # Honor user-provided gradient checkpointing preference (H200 speed-first profile)
        # Do not auto-enable at higher resolutions; caller can set low_vram/gradient_checkpointing as needed.
        
        # Create training configuration
        logger.info("Creating training configuration...")
        config = create_training_config(
            job_name=job_name,
            dataset_stats=dataset_stats,
            steps=steps,
            learning_rate=learning_rate,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            resolution=resolution,
            default_caption=default_caption,
            caption_dropout_rate=caption_dropout_rate,
            batch_size=batch_size,
            optimizer=optimizer,
            sample_every=sample_every,
            sample_prompts=prompts_list,
            gradient_checkpointing=gradient_checkpointing,
            low_vram=low_vram,
            use_ema=use_ema,
            ema_decay=ema_decay
        )
        
        # Add cache_latents_to_disk setting
        config["config"]["process"][0]["datasets"][0]["cache_latents_to_disk"] = cache_latents_to_disk
        
        # Log training parameters
        logger.info(f"Training parameters:")
        logger.info(f"  - Steps: {steps}")
        logger.info(f"  - Learning rate: {learning_rate}")
        logger.info(f"  - LoRA rank/alpha: {lora_rank}/{lora_alpha}")
        logger.info(f"  - Resolution: {resolution}")
        logger.info(f"  - Batch size: {batch_size}")
        logger.info(f"  - Optimizer: {optimizer}")
        logger.info(f"  - Gradient checkpointing: {gradient_checkpointing}")
        logger.info(f"  - Low VRAM: {low_vram}")
        logger.info(f"  - Cache latents: {cache_latents_to_disk}")
        
        # Run training
        logger.info("Starting training process...")
        run_training(config, job_name)
        
        # Create output archive
        logger.info("Creating output archive...")
        output_path = create_output_archive(job_name, dataset_stats)
        
        logger.info(f"Training completed successfully! Output: {output_path}")
        
        return TrainingOutput(weights=output_path)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        # Clean up on failure
        clean_up()
        raise
    
    finally:
        # Always clean up input directory
        if INPUT_DIR.exists():
            shutil.rmtree(INPUT_DIR)


if __name__ == "__main__":
    """
    For testing the trainer locally
    """
    print("Qwen Image LoRA Trainer")
    print("Use with: cog train -i dataset=@dataset.zip -i steps=2000 ...")
    print("Example: cog train -i dataset=@my_images.zip -i default_caption='A photo of my_subject' -i steps=1500")