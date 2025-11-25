#!/usr/bin/env python3
"""Qwen Image predictor with LoRA support."""

import hashlib
import os
import random
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

MODEL_CACHE_DIR = Path("model_cache")
LORA_CACHE_DIR = Path("/tmp/qwen_lora_cache")
BASE_URL = "https://weights.replicate.delivery/default/qwen-image-lora/model_cache/"

# Configure caches and CUDA behaviour before importing heavy libraries.
os.environ["HF_HOME"] = str(MODEL_CACHE_DIR)
os.environ["TORCH_HOME"] = str(MODEL_CACHE_DIR)
os.environ["HF_DATASETS_CACHE"] = str(MODEL_CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(MODEL_CACHE_DIR)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(MODEL_CACHE_DIR)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:512")
os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")

import sys

import torch
from cog import BasePredictor, Input, Path as CogPath
from safetensors import safe_open

sys.path.insert(0, "./ai-toolkit")
from extensions_built_in.diffusion_models.qwen_image import QwenImageModel
from toolkit.config_modules import ModelConfig
from toolkit.lora_special import LoRASpecialNetwork
from helpers.billing.metrics import record_billing_metric

QUALITY_DIMENSIONS: Dict[str, Tuple[int, int]] = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1136),
    "3:4": (1136, 1472),
    "3:2": (1536, 1024),
    "2:3": (1024, 1536),
}

SPEED_DIMENSIONS: Dict[str, Tuple[int, int]] = {
    "1:1": (1024, 1024),
    "16:9": (1024, 576),
    "9:16": (576, 1280),
    "4:3": (1024, 768),
    "3:4": (768, 1024),
    "3:2": (1152, 768),
    "2:3": (768, 1152),
}


def download_weights(url: str, dest: Path) -> None:
    """Fetch a model artifact using pget."""
    start = time.time()
    print(f"[download] {url} -> {dest}")
    if dest.suffix == ".tar":
        dest = dest.parent
    command = ["pget", "-vf" + ("x" if url.endswith(".tar") else ""), url, str(dest)]
    subprocess.check_call(command, close_fds=False)
    print(f"[download] done in {time.time() - start:.2f}s")


def cache_key_for_path(path: Path) -> str:
    stat = path.stat()
    signature = f"{path.resolve()}::{stat.st_size}::{stat.st_mtime_ns}"
    return hashlib.sha1(signature.encode()).hexdigest()


def materialise_safetensors(source: Path) -> Path:
    """Return a safetensors file for the given source, extracting ZIPs on demand."""
    if source.suffix.lower() != ".zip":
        return source.resolve()

    cache_dir = LORA_CACHE_DIR / cache_key_for_path(source)
    cached_file = cache_dir / "lora.safetensors"
    if cached_file.exists():
        return cached_file

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(source, "r") as archive:
        members = [m for m in archive.namelist() if m.endswith(".safetensors")]
        if not members:
            raise FileNotFoundError("LoRA archive does not contain a .safetensors file")
        member = members[0]
        extracted = Path(archive.extract(member, path=cache_dir)).resolve()
        if extracted != cached_file:
            cached_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(extracted), cached_file)
            # Clean up empty folders that zipfile may have created.
            parent = extracted.parent
            while parent != cache_dir and not any(parent.iterdir()):
                tmp = parent
                parent = parent.parent
                tmp.rmdir()
    return cached_file


def inspect_lora(path: Path) -> Tuple[int, int]:
    """Return (rank, alpha) for a LoRA safetensors file."""
    with safe_open(path, framework="pt") as tensors:
        sample_key = next(k for k in tensors.keys() if ("lora_A" in k or "lora_down" in k))
        rank = tensors.get_tensor(sample_key).shape[0]
        alpha_key = sample_key.replace("lora_down", "alpha").replace("lora_A", "alpha")
        alpha = int(tensors.get_tensor(alpha_key).item()) if alpha_key in tensors.keys() else rank
    return rank, alpha


def choose_dimensions(aspect_ratio: str, image_size: str) -> Tuple[int, int]:
    table = QUALITY_DIMENSIONS if image_size == "optimize_for_quality" else SPEED_DIMENSIONS
    width, height = table.get(aspect_ratio, table["1:1"])
    width = (width // 16) * 16
    height = (height // 16) * 16
    return width, height


def load_image_tensor(path: Path, width: int, height: int) -> torch.Tensor:
    with Image.open(path) as img:
        image = img.convert("RGB")
        if image.size != (width, height):
            image = image.resize((width, height), Image.LANCZOS)
    array = np.array(image).astype("float32") / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    tensor = tensor * 2.0 - 1.0
    return tensor


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Initialise the base Qwen image pipeline once per container."""
        MODEL_CACHE_DIR.mkdir(exist_ok=True)
        LORA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        for filename in ("models--Qwen--Qwen-Image.tar", "xet.tar"):
            archive = MODEL_CACHE_DIR / filename
            target_dir = MODEL_CACHE_DIR / filename.replace(".tar", "")
            if not target_dir.exists():
                download_weights(BASE_URL + filename, archive)

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        cfg = ModelConfig(name_or_path="Qwen/Qwen-Image", arch="qwen_image", dtype="bf16")
        self.qwen = QwenImageModel(device=self.device, model_config=cfg, dtype=torch.bfloat16)
        self.qwen.load_model()
        self.pipe = self.qwen.get_generation_pipeline()

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            print(f"Loaded Qwen/Qwen-Image on {props.name} ({props.total_memory / 1024 ** 3:.1f} GB)")
        else:
            print("Loaded Qwen/Qwen-Image on CPU")

        self.lora_net: Optional[LoRASpecialNetwork] = None
        self._lora_meta_cache: Dict[Path, Tuple[int, int]] = {}
        self._active_lora_path: Optional[Path] = None

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(description="Prompt for generated image"),
        enhance_prompt: bool = Input(
            description="Append a high-detail suffix to the prompt.",
            default=False,
        ),
        lora_weights: Optional[str] = Input(
            description=(
                "Load LoRA weights. Supports local .safetensors paths or ZIPs produced by cog train."
            ),
            default=None,
        ),
        replicate_weights: Optional[CogPath] = Input(
            description=(
                "LoRA ZIP generated by cog train (alternate to lora_weights)."
            ),
            default=None,
        ),
        lora_scale: float = Input(
            description="Determines how strongly the loaded LoRA should be applied.",
            default=1.0,
        ),
        image: Optional[CogPath] = Input(
            description="Optional guide image for img2img.",
            default=None,
        ),
        strength: float = Input(
            description="Strength for img2img pipeline",
            default=0.9,
            ge=0.0,
            le=1.0,
        ),
        negative_prompt: str = Input(
            description="Negative prompt for generated image",
            default=" ",
        ),
        aspect_ratio: str = Input(
            description="Aspect ratio for the generated image",
            choices=list(QUALITY_DIMENSIONS.keys()),
            default="16:9",
        ),
        image_size: str = Input(
            description="Image size preset (quality = larger, speed = faster).",
            choices=["optimize_for_quality", "optimize_for_speed"],
            default="optimize_for_quality",
        ),
        go_fast: bool = Input(
            description="Run faster predictions with aggressive caching.",
            default=True,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps (1–50).",
            ge=1,
            le=50,
            default=30,
        ),
        guidance: float = Input(
            description="Guidance for generated image (0-10).",
            ge=0.0,
            le=10.0,
            default=3.0,
        ),
        seed: Optional[int] = Input(
            description="Random seed. Leave blank for random.",
            default=None,
        ),
        output_format: str = Input(
            description="Format of the output images",
            choices=["webp", "jpg", "png"],
            default="webp",
        ),
        output_quality: int = Input(
            description="Quality when saving lossy images (0-100).",
            default=80,
            ge=0,
            le=100,
        ),
        disable_safety_checker: bool = Input(
            description="Disable safety checker (not used).",
            default=False,
        ),
    ) -> List[CogPath]:
        if disable_safety_checker:
            print("Safety checker not integrated in this build; generate responsibly.")

        if lora_weights and replicate_weights:
            print("Both lora_weights and replicate_weights supplied; using replicate_weights and ignoring lora_weights.")
            lora_weights = None

        primary_lora_path: Optional[Path] = None
        if lora_weights:
            candidate = Path(lora_weights)
            if candidate.exists():
                primary_lora_path = candidate
            else:
                print(f"LoRA weights not found at {candidate}; continuing without them.")
        elif replicate_weights:
            primary_lora_path = Path(replicate_weights)

        if primary_lora_path is not None:
            lora_file = materialise_safetensors(primary_lora_path)
            meta = self._lora_meta_cache.get(lora_file)
            if meta is None:
                meta = inspect_lora(lora_file)
                self._lora_meta_cache[lora_file] = meta

            rank, alpha = meta
            if (
                self.lora_net is None
                or getattr(self.lora_net, "lora_dim", None) != rank
                or getattr(self.lora_net, "alpha", None) != alpha
            ):
                self.lora_net = LoRASpecialNetwork(
                    text_encoder=self.qwen.text_encoder,
                    unet=self.qwen.unet,
                    lora_dim=rank,
                    alpha=alpha,
                    multiplier=lora_scale,
                    train_unet=True,
                    train_text_encoder=False,
                    is_transformer=True,
                    transformer_only=True,
                    base_model=self.qwen,
                    target_lin_modules=["QwenImageTransformer2DModel"],
                )
                self.lora_net.apply_to(
                    self.qwen.text_encoder, self.qwen.unet, apply_text_encoder=False, apply_unet=True
                )
                self.lora_net.force_to(self.qwen.device_torch, dtype=self.qwen.torch_dtype)

            self.lora_net.load_weights(str(lora_file))
            self.lora_net.is_active = True
            self.lora_net.multiplier = lora_scale
            self.lora_net._update_torch_multiplier()
            self._active_lora_path = lora_file
            print(f"Loaded LoRA: rank={rank}, alpha={alpha}, scale={lora_scale}")
        elif self.lora_net is not None:
            self.lora_net.is_active = False
            self.lora_net._update_torch_multiplier()
            self._active_lora_path = None

        def snap_dim(value: int) -> int:
            value = max(512, min(2048, value))
            snapped = (value // 16) * 16
            return snapped if snapped >= 512 else 512

        chosen_width: int
        chosen_height: int
        image_path: Optional[Path] = None
        if image is not None:
            image_path = Path(str(image))
            with Image.open(image_path) as base_img:
                guide_width, guide_height = base_img.size
            chosen_width = snap_dim(guide_width)
            chosen_height = snap_dim(guide_height)
            if (guide_width, guide_height) != (chosen_width, chosen_height):
                print(
                    f"Resized guide from {guide_width}x{guide_height} to {chosen_width}x{chosen_height}"
                )
        else:
            chosen_width, chosen_height = choose_dimensions(aspect_ratio, image_size)
            image_path = None

        if go_fast and num_inference_steps > 28:
            num_inference_steps = 28

        actual_seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        print(
            f"Generating: seed={actual_seed}, size={chosen_width}x{chosen_height}, steps={num_inference_steps}, guidance={guidance}"
        )

        if enhance_prompt:
            prompt = f"{prompt}, highly detailed, crisp focus, studio lighting, photorealistic"

        generator = torch.Generator(device=self.qwen.device_torch).manual_seed(actual_seed)

        latents_override: Optional[torch.Tensor] = None
        if image_path is not None:
            strength = float(max(0.0, min(1.0, strength)))
            base_tensor = load_image_tensor(image_path, chosen_width, chosen_height)
            latents = self.qwen.encode_images([base_tensor]).to(self.qwen.device_torch, dtype=self.qwen.torch_dtype)
            if strength > 0:
                noise = torch.randn(latents.shape, device=latents.device, dtype=latents.dtype, generator=generator)
                latents_override = torch.lerp(latents, noise, strength)
            else:
                latents_override = latents
            print(f"Img2img enabled with strength={strength:.2f}")
        cond = self.qwen.get_prompt_embeds(prompt)
        uncond = self.qwen.get_prompt_embeds(negative_prompt if negative_prompt.strip() else "")

        render_request = type(
            "RenderRequest",
            (),
            {
                "width": chosen_width,
                "height": chosen_height,
                "guidance_scale": guidance,
                "num_inference_steps": num_inference_steps,
                "latents": None,
                "ctrl_img": None,
            },
        )()
        if latents_override is not None:
            render_request.latents = latents_override

        start = time.time()
        result_image = self.qwen.generate_single_image(
            self.pipe, render_request, cond, uncond, generator, extra={}
        )
        print(f"Render finished in {time.time() - start:.2f}s")

        output_path = Path("/tmp") / f"output-{int(time.time() * 1000)}.{output_format}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {"quality": output_quality} if output_format in {"jpg", "webp"} else {}
        if output_format == "jpg":
            save_kwargs["optimize"] = True
        result_image.save(output_path, **save_kwargs)

        record_billing_metric("image_output_count", 1)
        return [CogPath(str(output_path))]
