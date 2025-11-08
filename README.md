# Qwen Image LoRA Trainer

[![Run on Replicate](https://replicate.com/qwen/qwen-image-lora/badge)](https://replicate.com/qwen/qwen-image-lora)

Production-ready toolkit for fine-tuning and deploying [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) LoRAs. Optimised for Replicate's H100/H200 fleet, yet lightweight enough for local experimentation. Build stylistic LoRAs, character likenesses, and brand-specific generators with a workflow that indie hackers can understand and extend in minutes.

## Why this repo?

- **One-command fine-tuning** – `cog train` configures the ai-toolkit backend, converts LoRA keys for Pruna/FlashAttention, and packages a ready-to-share ZIP.
- **Battle-tested inference** – `cog predict` supports text-to-image and img2img, dynamic LoRA loading, and deterministic seeds while keeping the codebase approachable.
- **Hardware-aware defaults** – Automatically adapts batch sizes, resolution tiers, and gradient checkpointing based on available VRAM.
- **Hackable by design** – Clear helpers, minimal branching, and readable flow make it easy to add new schedulers, caches, or safety filters without a rewrite.

## Quickstart

Clone with submodules and install Cog:

```bash
git clone --recursive https://github.com/replicate/qwen-image-lora-trainer.git
cd qwen-image-lora-trainer
pip install cog
```

### 1. Train a LoRA

```bash
cog train \
  -i dataset=@path/to/dataset.zip \
  -i default_caption="A photo of <>"
```

What happens under the hood:

- Extracts the dataset, normalises captions, and auto-fills missing `.txt` files.
- Detects GPU VRAM to pick safe resolutions and gradient-checkpointing settings.
- Trains a rank-32 LoRA for 1,000 steps at a 5e-4 learning rate (tunable via inputs).
- Converts `lora.safetensors` into Pruna-compatible keys and zips it with config metadata.

Output: `/tmp/qwen_lora_<timestamp>_trained.zip` containing `lora.safetensors`, `config.yaml`, and `settings.txt`.

### 2. Run inference

```bash
cog predict \
  -i prompt="Studio portrait of <>, cinematic lighting" \
  -i replicate_weights=@/tmp/qwen_lora_123456789_trained.zip \
  -i output_format=webp
```

Want guided transformations? Add `-i image=@guide.png -i strength=0.6` for img2img. Set `-i go_fast=false` when chasing maximum fidelity.

## Predictor input reference

| Input | Description | Default |
|-------|-------------|---------|
| `prompt` | Primary text prompt | _required_ |
| `enhance_prompt` | Appends a high-detail suffix for sharper renders | `false` |
| `lora_weights` | Path/ZIP for LoRA weights (local paths preferred) | `null` |
| `replicate_weights` | ZIP emitted by `cog train`; overrides `lora_weights` when both are set | `null` |
| `lora_scale` | Multiplier for the loaded LoRA | `1.0` |
| `image` | Optional img2img guide (resized internally) | `null` |
| `strength` | Img2img blend factor (0 = copy guide, 1 = full noise) | `0.9` |
| `negative_prompt` | Concepts to avoid | `"(single space)"` |
| `aspect_ratio` | Resolution preset when no guide image is supplied | `16:9` |
| `image_size` | Quality vs speed profile | `optimize_for_quality` |
| `go_fast` | Aggressive caching + step clamp (~8 steps) | `true` |
| `num_inference_steps` | Diffusion steps (auto-clamped when `go_fast`) | `30` |
| `guidance` | Classifier-free guidance scale | `3.0` |
| `seed` | Deterministic seed (random when unset) | `null` |
| `output_format` | `webp`, `jpg`, or `png` | `webp` |
| `output_quality` | Quality for lossy formats | `80` |
| `disable_safety_checker` | Placeholder flag – prints a reminder only | `false` |

> LoRA ZIPs created by `cog train` can be fed directly into `replicate_weights`. The predictor extracts and caches the safetensors automatically.

## Dataset guidelines

Pack your dataset as a flat ZIP. Supported image formats: `.jpg`, `.jpeg`, `.png`, `.webp`.

```
my-dataset.zip
├── img001.jpg
├── img001.txt   # "A photo of <> wearing a navy hoodie"
├── img002.jpg
└── img003.jpg   # Falls back to default_caption
```

### Prompting best practices for Qwen Image

- Use literal, descriptive language. Qwen learns by overriding existing concepts, not inventing new tokens.
- Avoid placeholder handles like `TOK`, `sks`, or `zzz`. They actively hurt convergence.
- Keep captions grounded in real traits (clothing, lighting, scene) so inference prompts can remix them reliably.

## Training defaults & knobs

| Parameter | Default | Notes |
|-----------|---------|-------|
| `steps` | `1000` | Increase for larger datasets; saves occur at the final step. |
| `learning_rate` | `5e-4` | Balanced for portraits and style LoRAs. |
| `lora_rank` | `32` | Alpha matches rank; change for capacity vs size. |
| `batch_size` | `1` | Switch to `2` or `4` on high-VRAM GPUs. |
| `optimizer` | `adamw` | `adamw8bit`, `adam8bit`, and `prodigy` also available. |
| `seed` | random | Provide for reproducible fine-tunes. |

Training artefacts live under `output/<job_name>/` and are cleaned once the final ZIP is created.

## Advanced usage

- **Custom resolutions** – Img2img snaps the guide to multiples of 16. For text-to-image presets, adjust `QUALITY_DIMENSIONS` / `SPEED_DIMENSIONS` in `predict.py`.
- **LoRA hot swapping** – Metadata (rank/alpha) is cached per safetensors file so reloading LoRAs stays instant.
- **Extending safety** – Hook into `result_image` before saving if you want CLIP- or Falcon-based filters.
- **Local caching** – Model archives download to `model_cache/` once; LoRA ZIPs unpack to `/tmp/qwen_lora_cache` using a content hash.

## Troubleshooting

- **"LoRA weights not found"** – Check the path. The predictor logs a warning and continues with the base model when it cannot locate the file.
- **OOM during training** – Reduce `batch_size`, lower `steps`, or rely on the automatic resolution downgrade (A100 profile) when VRAM is limited.
- **Outputs look off** – Revisit your captions. Qwen Image rewards detailed, grounded captions that match your dataset.

## Contributing

Pull requests and custom integrations are welcome. The codebase purposely avoids heavy frameworks so you can:

- Swap in alternative schedulers or samplers.
- Add caching strategies for weights or latents.
- Layer on custom safety checkers or watermarking.

Tag releases with meaningful notes so downstream users know which defaults they depend on. Suggestions for better defaults, new dataset pipelines, or inference UX upgrades are always appreciated.

---

Happy fine-tuning! If you build something cool with this trainer, share it with the community—we're eager to see what you create.
