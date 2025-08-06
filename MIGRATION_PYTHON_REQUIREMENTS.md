# Python Requirements Migration Guide

## Summary of Changes

This project has been updated to use Cog's recommended `python_requirements` instead of the deprecated `python_packages` in `cog.yaml`.

## What Changed

### Before (Deprecated)
```yaml
build:
  python_packages:
    - "torch>=2.1.0"
    - "diffusers>=0.24.0"
    # ... many more packages
```

### After (Recommended)
```yaml
build:
  python_requirements: requirements.txt
```

## Benefits

1. **No More Deprecation Warnings**: The `python_packages` field was deprecated and would be removed in future Cog versions
2. **Better Version Control**: All dependencies are now in `requirements.txt` for better tracking and reproducibility
3. **Easier Maintenance**: Dependencies can be updated and tested independently
4. **Standard Practice**: Follows Python packaging best practices

## Cog Contract Validation

### Training Contract ✅
- **Function**: `train.py:train` - Top-level function that returns a single archive
- **Output**: Single `TrainingOutput(weights=Path)` containing `lora/lora_weights.safetensors` and `meta.json`
- **Flag-gated**: `train_placeholder_token: bool = Input(default=False)` properly gates zero-prior feature

### Prediction Contract ✅
- **Class**: `predict.py:Predictor` - Class inheriting from `BasePredictor`
- **Method**: `predict(...)` method with `replicate_weights: Path` parameter
- **Archive Support**: Extracts zip/tar archives and loads LoRA weights with metadata

## Zero-Prior Token Feature

The zero-prior placeholder token feature is properly implemented with:

1. **Default OFF**: `train_placeholder_token=false` by default
2. **When ON**:
   - Adds special token (e.g., `<zwx>`) to vocabulary
   - Resizes text encoder embeddings
   - Near-zero initialization of placeholder embedding
   - Gradient masking to only update placeholder token
   - TE quantization disabled when feature is active

3. **Validation**:
   - Single-token encoding verification
   - Metadata preservation for inference
   - Archive structure validation

## Usage Examples

### Baseline Training (Feature OFF)
```bash
cog train \
  -i dataset=@dataset.zip \
  -i train_placeholder_token=false \
  -i rank=32 \
  -i steps=50 \
  -i learning_rate=5e-5
```

### Zero-Prior Training (Feature ON)
```bash
cog train \
  -i dataset=@dataset.zip \
  -i train_placeholder_token=true \
  -i placeholder_token="<zwx>" \
  -i placeholder_init_std=1e-5 \
  -i rank=32 \
  -i steps=50 \
  -i learning_rate=5e-5
```

### Prediction with Both Modes
```bash
cog predict \
  -i prompt="a photo of <zwx>" \
  -i steps=20 \
  -i guidance=4 \
  -i width=1024 \
  -i height=1024 \
  -i seed=42 \
  -i lora_scale=1.0 \
  -i replicate_weights=@weights.zip
```

## Technical Notes

1. **Archive Format**: Both training modes produce identical archive structure:
   ```
   weights.zip
   ├── lora/
   │   └── lora_weights.safetensors
   └── meta.json
   ```

2. **Metadata**: Includes training configuration and placeholder token information:
   ```json
   {
     "base_model": "Qwen/Qwen-Image",
     "train_placeholder_token": true/false,
     "placeholder_token": "<zwx>" | null,
     "placeholder_token_id": 151665 | null,
     "rank": 32,
     "training_steps": 50
   }
   ```

3. **Dependency Management**: The `requirements.txt` uses compatible version ranges to avoid conflicts

## Verification Steps

The implementation has been validated for:
- ✅ Cog YAML deprecation warning resolved
- ✅ Training function contract (`train.py:train`)
- ✅ Prediction class contract (`predict.py:Predictor`)  
- ✅ Zero-prior token feature properly flag-gated
- ✅ Archive structure and metadata preservation
- ✅ Single-token encoding verification

## Migration Impact

- **Breaking Changes**: None - existing functionality preserved
- **Performance**: No performance impact
- **Build Times**: Potentially faster with optimized dependency resolution
- **Compatibility**: Full backward compatibility with existing archives

---

**Note**: This migration ensures the project follows Cog's current best practices and will continue to work with future Cog versions.