#!/usr/bin/env python3
"""Standalone helper to load a Qwen Image LoRA and render a single sample."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import torch
from safetensors.torch import load_file

# Make the ai-toolkit submodule importable before bringing in its modules.
sys.path.insert(0, "./ai-toolkit")

from extensions_built_in.diffusion_models.qwen_image import QwenImageModel
from toolkit.config_modules import ModelConfig
from toolkit.lora_special import LoRASpecialNetwork

LORA_PATH = Path("qwen_lora_v1.safetensors")
PROMPT = "a photo of a man named zeke"
OUTPUT_PATH = Path("zeke_with_lora.png")
SEED = 42


def find_rank_and_alpha(weights: dict[str, torch.Tensor]) -> Tuple[int, int]:
    """Infer LoRA rank/alpha from one of the A/down tensors."""
    key = next(k for k in weights if "lora_A" in k or "lora_down" in k)
    rank = weights[key].shape[0]
    alpha_key = key.replace("lora_down", "alpha").replace("lora_A", "alpha")
    alpha = int(weights[alpha_key].item()) if alpha_key in weights else rank
    return rank, alpha


def attach_lora(qwen: QwenImageModel, rank: int, alpha: int) -> LoRASpecialNetwork:
    """Create and connect a LoRA network to the Qwen transformer."""
    net = LoRASpecialNetwork(
        text_encoder=qwen.text_encoder,
        unet=qwen.unet,
        lora_dim=rank,
        alpha=alpha,
        multiplier=1.0,
        train_unet=True,
        train_text_encoder=False,
        is_transformer=True,
        transformer_only=True,
        base_model=qwen,
        target_lin_modules=["QwenImageTransformer2DModel"],
    )
    net.apply_to(qwen.text_encoder, qwen.unet, apply_text_encoder=False, apply_unet=True)
    net.force_to(qwen.device_torch, dtype=qwen.torch_dtype)
    net.eval()
    return net


def main() -> None:
    if not LORA_PATH.exists():
        raise SystemExit(f"LoRA file not found: {LORA_PATH}")

    model_cfg = ModelConfig(name_or_path="Qwen/Qwen-Image", arch="qwen_image", dtype="bf16")
    qwen = QwenImageModel(device="cuda:0" if torch.cuda.is_available() else "cpu",
                          model_config=model_cfg,
                          dtype=torch.bfloat16)
    qwen.load_model()

    tensors = load_file(str(LORA_PATH))
    rank, alpha = find_rank_and_alpha(tensors)
    lora_net = attach_lora(qwen, rank, alpha)
    lora_net.load_weights(str(LORA_PATH))
    lora_net.is_active = True
    lora_net._update_torch_multiplier()

    pipe = qwen.get_generation_pipeline()
    generator = torch.Generator(device=qwen.device_torch).manual_seed(SEED)
    cond = qwen.get_prompt_embeds(PROMPT)
    uncond = qwen.get_prompt_embeds("")
    render_request = SimpleNamespace(
        width=1024,
        height=1024,
        guidance_scale=4.0,
        num_inference_steps=20,
        latents=None,
        ctrl_img=None,
    )
    image = qwen.generate_single_image(pipe, render_request, cond, uncond, generator, extra={})
    image.save(OUTPUT_PATH)
    print(f"Wrote {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
