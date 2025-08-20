# Qwen Image LoRA Trainer

A Cog-powered implementation for training and running inference with Qwen Image models and LoRA adapters.

## Features

- **LoRA Training**: Fine-tune Qwen Image models with LoRA adapters
- **High-Resolution Generation**: Support for images up to 1664x1536 pixels
- **Multiple Aspect Ratios**: 1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3
- **Optimized Presets**: 
  - `optimize_for_quality`: ~1.5-1.7 megapixels for high-quality outputs
  - `optimize_for_speed`: ~0.7-0.8 megapixels for faster generation
- **Automatic Seed Logging**: Seeds are logged when randomly generated for reproducibility

## Setup

1. Clone the repository with submodules:
   ```bash
   git clone --recursive <repo-url>
   # Or if already cloned:
   git submodule update --init --recursive
   ```

2. Install dependencies:
   ```bash
   pip install -r ai-toolkit/requirements.txt
   ```

3. Prepare your dataset:
   ```bash
   unzip your_dataset.zip
   # Remove any existing .txt caption files if needed
   ```

## Training

1. Configure training parameters in `qwen_config.yaml`:
   - Adjust `dataset.folder_path` to point to your dataset
   - Modify training hyperparameters as needed

2. Launch training:
   ```bash
   cd ai-toolkit
   python run.py ../qwen_config.yaml
   ```

## Inference

### Using Cog (Recommended)

```bash
cog predict -i prompt="A beautiful sunset over mountains" \
            -i aspect_ratio="16:9" \
            -i image_size="optimize_for_quality" \
            -i seed=42
```

### Using Python Script

1. Update `LORA_FILE_PATH` in `qwen_lora_inference.py` to point to your LoRA weights
2. Run: `python qwen_lora_inference.py`

## Image Dimensions

The model automatically selects appropriate dimensions based on aspect ratio and quality preset:

### Quality Mode (optimize_for_quality)
- **1:1**: 1328x1328 (1.76 MP)
- **16:9**: 1664x928 (1.54 MP)
- **9:16**: 928x1664 (1.54 MP)
- **4:3**: 1472x1136 (1.67 MP)
- **3:4**: 1136x1472 (1.67 MP)
- **3:2**: 1536x1024 (1.57 MP)
- **2:3**: 1024x1536 (1.57 MP)

### Speed Mode (optimize_for_speed)
- **1:1**: 896x896 (0.80 MP)
- **16:9**: 1120x624 (0.70 MP)
- **9:16**: 624x1120 (0.70 MP)
- **4:3**: 992x768 (0.76 MP)
- **3:4**: 768x992 (0.76 MP)
- **3:2**: 1024x688 (0.70 MP)
- **2:3**: 688x1024 (0.70 MP)

All dimensions are automatically adjusted to be divisible by 16 (model requirement).

## API Parameters

- `prompt`: Text description of the image to generate
- `enhance_prompt`: Add quality-enhancing keywords to prompt (default: false)
- `negative_prompt`: Things to avoid in the generation
- `aspect_ratio`: Image aspect ratio (default: "16:9")
- `image_size`: Quality preset (default: "optimize_for_quality")
- `go_fast`: Enable speed optimizations (default: false)
- `num_inference_steps`: Denoising steps, 1-50 (default: 50)
- `guidance`: Guidance scale, 0-10 (default: 4.0)
- `seed`: Random seed for reproducibility (default: random)
- `output_format`: Image format - webp, jpg, png (default: "webp")
- `output_quality`: JPEG/WebP quality, 0-100 (default: 80)
- `replicate_weights`: Path to LoRA weights file
- `lora_scale`: LoRA influence strength, 0-3 (default: 1.0)

## Recent Updates

- **Increased image sizes**: Now supports up to 1664x1536 pixels matching production defaults
- **Seed logging**: Random seeds are now printed in logs for reproducibility
- **New aspect ratios**: Added 3:2 and 2:3 aspect ratio support
- **Improved dimension handling**: Automatic adjustment to ensure 16-pixel divisibility

## License

[Include your license information here]
