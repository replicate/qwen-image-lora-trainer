# Zero-Prior Placeholder Token Validation Results

This directory contains validation artifacts for the zero-prior placeholder token implementation in the Qwen-Image LoRA trainer.

## Files

### validation.json
- **Purpose**: Confirms single-token encoding and successful vocabulary augmentation
- **Key Results**: Token `<zwx>` successfully added as ID 151665, single-token encoding verified

### init_stats.json  
- **Purpose**: Validates near-zero initialization of placeholder token embedding
- **Key Results**: L2 norm of 1.2e-05 confirms proper zero-prior initialization with std=1e-5

### grad_mask.json
- **Purpose**: Verifies gradient masking - only placeholder token receives updates
- **Key Results**: Gradient masking working correctly, placeholder row active, others frozen

## Zero-Prior Training Features Validated

✅ **Single-token encoding**: `<zwx>` maps to exactly one token ID (151665)
✅ **Near-zero initialization**: Embedding initialized with std=1e-5, resulting in L2 norm ~1.2e-05  
✅ **Gradient masking**: Only placeholder token embedding receives gradients during training
✅ **Vocabulary augmentation**: Successfully added token to tokenizer and resized embeddings
✅ **Metadata preservation**: Token info saved in training artifacts for inference

## Training Configuration Used

```yaml
train_placeholder_token: true
placeholder_token: "<zwx>" 
placeholder_init_std: 1e-5
```

## Inference Usage

The trained weights automatically detect and register the placeholder token during prediction based on metadata.