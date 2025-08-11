#!/usr/bin/env python3
import os, sys, torch
from safetensors.torch import load_file

# Make toolkit importable
sys.path.insert(0, "./ai-toolkit")

from extensions_built_in.diffusion_models.qwen_image import QwenImageModel
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.config_modules import ModelConfig

# Adjust this
LORA_FILE_PATH = "/path/to/qwen_lora_v1.safetensors"

def main():
  device = "cuda:0"
  torch_dtype = torch.bfloat16


  # 2) Load base Qwen model
  model_cfg = ModelConfig(name_or_path="Qwen/Qwen-Image", arch="qwen_image", dtype="bf16")
  qwen = QwenImageModel(device=device, model_config=model_cfg, dtype=torch_dtype)
  qwen.load_model()

  # 3) Detect LoRA rank/alpha
  sd = load_file(LORA_FILE_PATH)
  sample_key = next(k for k in sd.keys() if ("lora_A" in k or "lora_down" in k))
  lora_dim = sd[sample_key].shape[0]
  alpha_key = sample_key.replace("lora_down", "alpha").replace("lora_A", "alpha")
  lora_alpha = int(sd[alpha_key].item()) if alpha_key in sd else lora_dim

  # 4) Build and apply LoRA network (transformer-only)
  lora_net = LoRASpecialNetwork(
    text_encoder=qwen.text_encoder,
    unet=qwen.unet,                 # alias to the Qwen transformer
    lora_dim=lora_dim,
    alpha=lora_alpha,
    multiplier=1.0,
    train_unet=True,
    train_text_encoder=False,
    is_transformer=True,
    transformer_only=True,
    base_model=qwen,
    # Qwen uses QwenImageTransformer2DModel as the target module class
    target_lin_modules=["QwenImageTransformer2DModel"]
  )
  lora_net.apply_to(qwen.text_encoder, qwen.unet, apply_text_encoder=False, apply_unet=True)
  lora_net.force_to(qwen.device_torch, dtype=qwen.torch_dtype)
  lora_net.eval()

  # 5) Load LoRA weights and activate
  lora_net.load_weights(LORA_FILE_PATH)
  lora_net.is_active = True
  lora_net._update_torch_multiplier()

  # 6) Build generation pipeline
  pipe = qwen.get_generation_pipeline()

  # 7) Generate
  prompt = "a photo of zeke sitting in a chair"
  gen_cfg = type("Gen", (), dict(width=1024, height=1024, guidance_scale=4.0,
                                 num_inference_steps=20, latents=None, ctrl_img=None))()
  g = torch.Generator(device=qwen.device_torch).manual_seed(42)
  cond = qwen.get_prompt_embeds(prompt)
  uncond = qwen.get_prompt_embeds("")
  img = qwen.generate_single_image(pipe, gen_cfg, cond, uncond, g, extra={})

  # 8) Save
  img.save("zeke.png")

if __name__ == "__main__":
  main()