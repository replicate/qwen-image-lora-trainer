# Qwen-Image LoRA Inference for Replicate

Simple inference package for running Qwen-Image with LoRA adapters and optional zero-prior placeholder tokens.

## Features

- Load base Qwen-Image model from Hugging Face
- Apply LoRA adapters trained with the ai-toolkit
- Support for placeholder tokens (like `<zwx>`) trained with zero-prior initialization
- Simple command-line interface
- Configurable generation parameters

## Installation

```bash
pip install torch diffusers transformers safetensors pillow
```

## Quick Start

### Basic usage (no LoRA):
```bash
python predict.py --prompt "a beautiful landscape" --output_path landscape.jpg
```

### With LoRA weights:
```bash
python predict.py \
    --prompt "a photo of a person in a forest" \
    --lora_path ./output/transformer/lora_weights.safetensors \
    --lora_scale 1.0 \
    --output_path forest_person.jpg
```

### With placeholder token and LoRA:
```bash
python predict.py \
    --prompt "a photo of <zwx> in a modern office" \
    --lora_path ./output/transformer/lora_weights.safetensors \
    --placeholder_token "<zwx>" \
    --lora_scale 1.0 \
    --output_path office_subject.jpg
```

## Command-line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--prompt` | Required | Text prompt for image generation |
| `--negative_prompt` | "" | Negative prompt to avoid certain features |
| `--lora_path` | None | Path to LoRA weights (.safetensors file) |
| `--lora_scale` | 1.0 | LoRA strength multiplier (0.0-2.0) |
| `--placeholder_token` | None | Special token (e.g., `<zwx>`) for trained subjects |
| `--width` | 1024 | Output image width |
| `--height` | 1024 | Output image height |
| `--steps` | 20 | Number of inference steps |
| `--guidance` | 4.0 | Guidance scale for prompt adherence |
| `--seed` | Random | Random seed for reproducibility |
| `--output_path` | generated_image.jpg | Output image file path |
| `--base_model` | "Qwen/Qwen-Image" | Base model path or HF model name |
| `--device` | "cuda" | Device to use (cuda/cpu) |
| `--dtype` | "bf16" | Data type (bf16/fp16/fp32) |

## Integration with Training

This inference package is designed to work with LoRA weights trained using the ai-toolkit trainer with the zero-prior placeholder token feature:

### Training Configuration
```yaml
train:
  train_placeholder_token: true    # Enable placeholder token training
  placeholder_token: "<zwx>"       # Use angle brackets for single-token encoding
  placeholder_init_std: 1e-5       # Near-zero initialization

sample:
  prompts:
    - "a photo of <zwx>"           # Use placeholder token in prompts
```

### After Training
1. Locate your trained LoRA weights (usually in `output/transformer/`)
2. Use the same `placeholder_token` value in inference
3. Include the placeholder token in your prompts

## Python API Usage

```python
from predict import QwenImageLoRAInference

# Initialize inference
inference = QwenImageLoRAInference(device="cuda", dtype="bf16")

# Load base model
inference.load_base_model()

# Register placeholder token (if used during training)
inference.register_placeholder_token("<zwx>")

# Load LoRA weights
inference.load_lora_weights("./output/transformer/lora_weights.safetensors")

# Generate image
image = inference.generate_image(
    prompt="a photo of <zwx> at the beach",
    width=1024,
    height=1024,
    num_inference_steps=20,
    guidance_scale=4.0,
    seed=42
)

# Save result
image.save("beach_photo.jpg")
```

## Troubleshooting

### Token Encoding Issues
If you see warnings about multi-token encoding:
```
⚠ WARNING: Token zwx encodes to 2 tokens: [123, 456]
```

**Solution**: Use angle brackets: `<zwx>` instead of `zwx` to ensure single-token encoding.

### Memory Issues
For systems with limited VRAM:
- Use `--dtype fp16` or `--dtype fp32`
- Reduce image dimensions: `--width 512 --height 512`
- Use CPU inference: `--device cpu`

### LoRA Loading Errors
- Ensure the LoRA path points to the correct `.safetensors` file
- Check that the LoRA was trained for the image transformer (not text encoder)
- Verify the LoRA is compatible with the base model version

## Replicate Deployment

To deploy on Replicate, create a `cog.yaml` configuration:

```yaml
build:
  python_version: "3.11"
  system_packages:
    - "libgl1-mesa-glx"
    - "libglib2.0-0"
  python_packages:
    - "torch>=2.0.0"
    - "diffusers>=0.24.0"
    - "transformers>=4.35.0"
    - "safetensors"
    - "pillow"

predict: "predict.py:QwenImageLoRAInference"
```

The inference class can be easily adapted for Cog's prediction interface by implementing the required `predict()` method.

## Performance Notes

- First inference run will be slower due to model loading
- Subsequent generations are much faster (model stays in memory)
- BF16 provides the best speed/quality tradeoff on modern GPUs
- Batch generation can be implemented for multiple images per request

## License

This inference package follows the same license as the base Qwen-Image model and ai-toolkit.