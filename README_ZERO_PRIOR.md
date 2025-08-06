# Zero-Prior Placeholder Token Training for Qwen-Image

## Overview

This implementation adds **opt-in zero-prior placeholder token training** to eliminate unwanted semantic biases from pre-trained text encoders when fine-tuning Qwen-Image models.

### The Problem

Traditional trigger words like `TOK`, `zwx`, or similar carry strong semantic priors from pre-training that can bias generation. For example:
- `TOK` might generate waffles regardless of your training data
- `zwx` might have unexpected associations from the base model

### The Solution

Zero-prior placeholder token training:

1. **Adds a fresh special token** (e.g., `<zwx>`) with near-zero initialization
2. **Freezes the entire text encoder** except for the new token embedding
3. **Uses gradient masking** to ensure only the placeholder token receives updates
4. **Keeps LoRA training on the image transformer** for efficiency

## Configuration

The feature is **OFF by default** and requires explicit activation:

### Training Config (`train` section)

```yaml
train:
  # Zero-prior placeholder token (opt-in, default: false)
  train_placeholder_token: true
  placeholder_token: "<zwx>"           # Use angle brackets for single-token encoding
  placeholder_init_std: 1e-5           # Near-zero initialization standard deviation
```

### Debug Config (`debug` section)

```yaml
debug:
  log_prompt_ids: true                 # Log token IDs during prompt processing
```

## Cog Interface

### Training (`cog train`)

Returns a single archive containing LoRA weights and metadata:

```bash
cog train \
  -i dataset=@dataset_zip_with_image_caption_pairs.zip \
  -i rank=32 \
  -i train_placeholder_token=true \
  -i placeholder_token="<zwx>" \
  -i steps=4000
```

**Output**: `weights.zip` containing:
```
weights.zip
├─ lora/
│  └─ lora_weights.safetensors
└─ meta.json  # {"base_model": "Qwen/Qwen-Image", "placeholder_token": "<zwx>", ...}
```

### Prediction (`cog predict`)

Consumes the training artifact and applies LoRA at inference:

```bash
cog predict \
  -i prompt="a photo of <zwx>" \
  -i steps=20 -i guidance=4 -i width=1024 -i height=1024 \
  -i lora_scale=1.0 \
  -i replicate_weights=@weights.zip
```

## Implementation Details

### Zero-Prior Token Setup

When `train_placeholder_token: true`:

1. **Token Addition**:
   ```python
   tokenizer.add_special_tokens({"additional_special_tokens": [placeholder_token]})
   ```

2. **Embedding Resize**:
   ```python
   text_encoder.resize_token_embeddings(len(tokenizer))
   ```

3. **Near-Zero Initialization**:
   ```python
   emb.weight[tok_id].normal_(mean=0.0, std=placeholder_init_std)
   ```

4. **Gradient Masking**:
   ```python
   def grad_mask_hook(grad):
       mask = torch.zeros_like(grad)
       mask[tok_id] = 1.0
       return grad * mask
   ```

### Archive Format

Training produces a single archive with:
- **`lora/lora_weights.safetensors`**: LoRA adapter weights
- **`meta.json`**: Metadata including placeholder token info

Sample metadata:
```json
{
  "base_model": "Qwen/Qwen-Image",
  "placeholder_token": "<zwx>",
  "placeholder_token_id": 151665,
  "rank": 32,
  "created": "2025-01-06T01:23:45Z"
}
```

### Inference Workflow

1. **Extract Archive**: Unzip/untar the weights file
2. **Load Metadata**: Parse configuration from `meta.json`
3. **Register Token**: Add placeholder token to tokenizer/text_encoder
4. **Apply LoRA**: Load adapters onto image transformer
5. **Generate**: Run inference with combined model

## Important Caveats

### Single Token Encoding

⚠️ **Use angle brackets** to ensure single-token encoding:
- ✅ `<zwx>` → Single token ID
- ❌ `zwx` → May split into multiple tokens

The system will warn if the token encodes to multiple IDs.

### Text Encoder Quantization

⚠️ **Do not quantize the text encoder** when training placeholder tokens:
```yaml
model:
  quantize_te: false  # Required for placeholder token training
```

### Optimizer Configuration

The placeholder token embedding is added to the optimizer automatically when the feature is enabled (with `weight_decay=0.0`).

## Diagnostics

Enable debug logging to see token processing:

```yaml
debug:
  log_prompt_ids: true
```

Sample output:
```
🏷️ Placeholder token '<zwx>' registered with ID: 151665
📊 Prompt 'a photo of <zwx>' tokenized to: [64, 6548, 315, 220, 151665]
🎯 Gradient mask active: only token 151665 receives updates
```

## Examples

### Basic Training

```bash
# Train with zero-prior token
cog train \
  -i dataset=@my_photos.zip \
  -i placeholder_token="<zwx>" \
  -i rank=64 \
  -i steps=2000
```

### Advanced Configuration

```yaml
train:
  train_placeholder_token: true
  placeholder_token: "<mytoken>"
  placeholder_init_std: 1e-6        # Even closer to zero
  
debug:
  log_prompt_ids: true              # Enable diagnostics
  
model:
  quantize: false                   # Full precision
  quantize_te: false               # Required for token training
```

### Inference

```bash
# Generate with trained model
cog predict \
  -i prompt="a portrait of <zwx> in golden hour light" \
  -i negative_prompt="blurry, low quality" \
  -i steps=25 \
  -i guidance_scale=4.5 \
  -i lora_scale=1.2 \
  -i replicate_weights=@my_trained_weights.zip
```

## Validation Results

Our implementation has been validated with comprehensive tests:

- ✅ **Placeholder Token Setup**: Correct token addition and ID assignment
- ✅ **Gradient Masking**: Only placeholder token receives gradients
- ✅ **Prompt Tokenization**: Token properly recognized in various contexts  
- ✅ **Archive Operations**: Weights packaging and extraction
- ✅ **Metadata Persistence**: Configuration preserved between train/predict

## Performance Notes

- **Memory**: Minimal overhead (one additional embedding row)
- **Speed**: No impact on inference speed
- **Storage**: Archive adds ~50MB for typical LoRA weights + minimal metadata

## Troubleshooting

### "Token encodes to multiple IDs"

**Problem**: Placeholder token splits during tokenization
**Solution**: Use angle brackets (e.g., `<zwx>` instead of `zwx`)

### "Text encoder quantization error"

**Problem**: Cannot update embeddings on quantized model
**Solution**: Set `quantize_te: false` in model config

### "LoRA weights not found"

**Problem**: Archive structure mismatch
**Solution**: Check that training completed and produced `lora/lora_weights.safetensors`

## Advanced Usage

### Custom Initialization

```yaml
train:
  placeholder_init_std: 0.0         # Perfect zero initialization
```

### Multiple Tokens

While not officially supported, you can train multiple placeholder tokens by running separate training sessions.

### Integration with Existing LoRA

The zero-prior tokens can be combined with traditional LoRA training - the text encoder gets the placeholder token while the image transformer gets LoRA adapters.