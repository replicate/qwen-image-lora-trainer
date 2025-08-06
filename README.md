# Qwen-Image LoRA Training + Zero-Prior Placeholder Tokens

This repository provides **LoRA fine-tuning** for Qwen-Image with an optional **zero-prior placeholder token** feature to eliminate unwanted training biases, plus **LoRA inference** for Replicate deployment.

## 🚀 Quick Start

### Basic Training
```bash
# Install ai-toolkit dependencies
cd ai-toolkit && pip install -r requirements.txt

# Launch training
python run.py ../qwen_config.yaml
```

### Training with Zero-Prior Placeholder Token (Recommended)
```bash
# Use the enhanced config with placeholder token training
python run.py ../qwen_config_placeholder.yaml
```

## 🎯 Features

### ✨ Zero-Prior Placeholder Token Training (NEW)
- **Problem**: Regular trigger words like `TOK` or `zwx` carry unwanted semantic priors from pre-training
- **Solution**: Add a brand-new special token (e.g., `<zwx>`) with near-zero initialization
- **Result**: Clean, unbiased training that learns only from your dataset

### 🔧 LoRA Inference for Replicate
- Simple Python package for loading base + LoRA models
- Support for placeholder tokens in inference
- Command-line and programmatic interfaces
- Ready for Replicate deployment

### 📊 Debug & Diagnostics
- Optional prompt tokenization logging
- Gradient masking verification
- System prompt detection

## 📋 Configuration

### Standard Configuration (qwen_config.yaml)
```yaml
train:
  train_unet: true
  train_text_encoder: false
  # ... other settings

sample:
  prompts:
    - "a photo of zwx"  # Traditional approach - may have bias
```

### Enhanced Configuration with Placeholder Token
```yaml
train:
  train_unet: true
  train_text_encoder: false
  
  # Zero-prior placeholder token feature (opt-in)
  train_placeholder_token: true      # Enable the feature
  placeholder_token: "<zwx>"         # Use angle brackets for single token
  placeholder_init_std: 1e-5         # Near-zero initialization

debug:
  log_prompt_ids: false              # Enable for tokenization debugging

sample:
  prompts:
    - "a photo of <zwx>"             # Clean, unbiased training
```

## 🛠️ How It Works

### Training Process
1. **Token Addition**: Adds `<zwx>` as a new special token to the tokenizer
2. **Embedding Resize**: Expands text encoder embedding matrix
3. **Zero Initialization**: Sets the new token's embedding to ~0 (configurable std)
4. **Selective Training**: Only the placeholder token embedding receives gradients
5. **LoRA on Image Side**: Regular LoRA training continues on the image transformer

### Key Benefits
- **No semantic bias** from pre-trained token meanings
- **Faster convergence** with cleaner gradients
- **Better prompt adherence** without fighting existing associations
- **Backward compatible** - defaults to OFF

## 🧪 Validation & Testing

Run smoke tests to verify functionality:
```bash
python test_placeholder_token.py
```

This tests:
- Placeholder token setup and initialization
- Gradient masking (only placeholder token receives gradients)
- Prompt tokenization with the new token

## 🚀 Inference

### Cog Workflow (Production - Recommended)

**Training:**
```bash
# Train with Cog (creates weights.zip archive)
cog train \
    -i dataset=@training_images.zip \
    -i placeholder_token="<zwx>" \
    -i rank=64 \
    -i steps=4000 \
    -i learning_rate=1e-4

# This creates weights.zip containing LoRA + metadata
```

**Inference:**
```bash
# Generate with trained model
cog predict \
    -i prompt="a photo of <zwx> in a modern office" \
    -i replicate_weights=@weights.zip \
    -i width=1024 \
    -i height=1024 \
    -i guidance_scale=4.0

# Without custom weights (base model only)  
cog predict -i prompt="a beautiful landscape"
```

### Local Development & Testing

**Direct Python API:**
```python
# Training
from train import train
from cog import Path

result = train(
    dataset=Path("training_images.zip"),
    placeholder_token="<zwx>",
    rank=64,
    steps=4000
)
print(f"Weights saved to: {result.weights}")

# Inference  
from predict import Predictor

predictor = Predictor()
predictor.setup()

image_path = predictor.predict(
    prompt="a photo of <zwx> at the beach",
    replicate_weights=result.weights,
    guidance_scale=4.0
)
print(f"Generated: {image_path}")
```

**Legacy Inference (Standalone):**
```bash
cd replicate_infer
python predict.py \
    --prompt "a photo of <zwx>" \
    --lora_path ../output/transformer/lora_weights.safetensors \
    --placeholder_token "<zwx>"
```

See [replicate_infer/README.md](replicate_infer/README.md) for legacy standalone inference.

## ⚙️ Configuration Reference

### Training Options
| Option | Default | Description |
|--------|---------|-------------|
| `train_placeholder_token` | `false` | Enable zero-prior placeholder token training |
| `placeholder_token` | `"<zwx>"` | The special token to add (use angle brackets) |
| `placeholder_init_std` | `1e-5` | Initialization standard deviation (0.0 for exact zeros) |

### Debug Options
| Option | Default | Description |
|--------|---------|-------------|
| `log_prompt_ids` | `false` | Log tokenization details and system prompt detection |

## 🔍 Troubleshooting

### Token Encoding Issues
**Problem**: Warning about multi-token encoding
```
WARNING: Token zwx encodes to 2 tokens: [123, 456]
```
**Solution**: Use angle brackets: `<zwx>` instead of `zwx`

### Text Encoder Quantization
**Problem**: Placeholder token training with quantized text encoder
**Solution**: The system automatically disables TE quantization when placeholder training is enabled

### Memory Issues
- Placeholder token training adds minimal memory overhead
- Use standard LoRA memory optimization techniques
- Disable debug logging in production

## 🎨 Examples

### Training Comparison

**Before (traditional trigger):**
```yaml
sample:
  prompts: ["a photo of TOK"]
# Result: Images of waffles due to pre-trained bias
```

**After (placeholder token):**
```yaml
train:
  train_placeholder_token: true
  placeholder_token: "<zwx>"
sample:
  prompts: ["a photo of <zwx>"]
# Result: Clean learning from your training data
```

### Inference Usage
```bash
# Traditional approach - may have bias from training
python predict.py --prompt "a photo of zwx in a forest"

# Zero-prior approach - clean results
python predict.py --prompt "a photo of <zwx> in a forest" --placeholder_token "<zwx>"
```

## 📁 Repository Structure

```
├── cog.yaml                      # Cog configuration for Replicate deployment
├── train.py                      # Cog training interface 
├── predict.py                    # Cog prediction interface
├── qwen_config.yaml              # Standard training config (legacy)
├── qwen_config_placeholder.yaml  # Enhanced config with placeholder tokens
├── test_placeholder_token.py     # Comprehensive test suite
├── replicate_infer/              # Legacy standalone inference package
│   ├── predict.py               # Standalone inference script
│   ├── example.py               # Usage examples  
│   └── README.md                # Standalone inference docs
└── ai-toolkit/                   # Submodule with enhanced training code
    ├── toolkit/config_modules.py # Added placeholder token configs
    ├── jobs/process/BaseSDTrainProcess.py # Enhanced to pass train_config
    └── extensions_built_in/diffusion_models/qwen_image/
        └── qwen_image.py         # Core placeholder token implementation
```

## 🔄 Migration Guide

### From Legacy to Cog Workflow

**Before (legacy standalone):**
```bash
# Training
cd ai-toolkit && python run.py ../qwen_config_placeholder.yaml

# Inference  
cd ../replicate_infer
python predict.py --prompt "a photo of <zwx>" --lora_path ../output/...
```

**After (Cog production):**
```bash
# Training
cog train -i dataset=@images.zip -i placeholder_token="<zwx>"

# Inference
cog predict -i prompt="a photo of <zwx>" -i replicate_weights=@weights.zip
```

**Benefits of Migration:**
- ✅ **Single archive**: No more managing separate LoRA files and configs
- ✅ **Automatic metadata**: Placeholder token info travels with weights  
- ✅ **Replicate ready**: Direct deployment without additional packaging
- ✅ **Version control**: Each training run produces a complete, reproducible artifact

## 🤝 Contributing

This implementation follows the design specified in the original issue. Key components:

- **Zero regression**: Feature is OFF by default
- **Minimal changes**: Focused implementation in QwenImageModel
- **Debug support**: Optional logging for transparency
- **Production ready**: Includes inference package and documentation

## 📄 License

This project follows the licensing of the base Qwen-Image model and ai-toolkit components.
