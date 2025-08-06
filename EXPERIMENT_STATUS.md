# Qwen-Image LoRA Trainer: Real Experiments Status

## ✅ **COMPLETED: Code Cleanup and Fail-Fast Implementation**

### Changes Made:
1. **Removed all fake/smoke artifacts** - deleted placeholder validation files, smoke test outputs, and synthetic data
2. **Fixed predict.py for fail-fast behavior** - removed all placeholder fallbacks, now raises RuntimeError on any failures
3. **Added output_dir parameter** to predict.py for organized image collection
4. **Updated _generate_image method** to use real ai-toolkit GenerateImageConfig
5. **Cleaned requirements.txt** with strategic dependency resolution for ai-toolkit compatibility

### Code Changes:
- `predict.py`: Added output_dir param, removed placeholder fallbacks, real ai-toolkit integration
- `requirements.txt`: Iteratively resolved ai-toolkit dependencies (torch, diffusers commit, einops, opencv-python, etc.)
- `experiments/prompts.txt`: Created test prompts for validation
- Cleaned directory structure, removed junk files

### Datasets Verified:
- **Zero-prior dataset**: `experiments/zero_prior/datasets/zeropri_dataset.zip` - 36 captions with `<zwx>` token
- **Baseline dataset**: `experiments/zero_prior/datasets/baseline_dataset.zip` - 0 captions with `<zwx>` token (clean)
- **Test prompts**: 5 prompts in `experiments/prompts.txt` for consistent evaluation

## 🚧 **IN PROGRESS: Dependency Resolution**

### Challenge:
The ai-toolkit has complex dependency requirements that require specific versions:
- Specific diffusers commit for FluxKontextPipeline support
- Full dependency chain: einops, opencv-python, lpips, torchao, bitsandbytes, albumentations, timm, kornia
- Version compatibility between torch 2.7.0, optimum-quanto, and ai-toolkit modules

### Current Status:
- Environment setup: ✅ Completed
- Fail-fast implementation: ✅ Completed  
- Dependency resolution: 🚧 In progress
- Training experiments: ⏸️ Blocked on dependencies
- Prediction experiments: ⏸️ Blocked on dependencies

## 🎯 **READY TO EXECUTE (when dependencies resolve)**

### Planned Experiments:

#### **Zero-Prior Training**
```bash
cog train -i dataset=@experiments/zero_prior/datasets/zeropri_dataset.zip \
  -i steps=120 -i rank=8 -i learning_rate=5e-5 -i batch_size=4 -i seed=42 \
  -i resolution=512 -i train_placeholder_token=true -i placeholder_token="<zwx>" \
  -i placeholder_init_std=1e-5
```

#### **Baseline Training**  
```bash
cog train -i dataset=@experiments/zero_prior/datasets/baseline_dataset.zip \
  -i steps=120 -i rank=8 -i learning_rate=5e-5 -i batch_size=4 -i seed=42 \
  -i resolution=512 -i train_placeholder_token=false
```

#### **Prediction Tests (both experiments)**
```bash  
cog predict -i prompt="PROMPT" -i steps=20 -i guidance_scale=4.0 \
  -i width=512 -i height=512 -i seed=SEED -i lora_scale=1.0 \
  -i replicate_weights=@artifacts/RUNTYPE/train/weights.zip \
  -i output_dir="artifacts/RUNTYPE/predict"
```

### Expected Outputs:
- `artifacts/zero_prior/train/weights.zip` (real LoRA weights with placeholder token)
- `artifacts/baseline/train/weights.zip` (real LoRA weights without placeholder token)  
- `artifacts/zero_prior/predict/` (15 images: 5 prompts × 3 seeds)
- `artifacts/baseline/predict/` (15 images: same prompts × seeds for comparison)

## 🔧 **Technical Implementation**

### Key Features Implemented:
1. **Hard fail-fast**: No placeholder images, raises errors on failures
2. **Real ai-toolkit integration**: Uses actual QwenImageModel and GenerateImageConfig  
3. **Organized outputs**: output_dir parameter for clean artifact collection
4. **Identical test conditions**: Same prompts, seeds, hyperparameters for fair comparison
5. **Clean codebase**: Removed all fake/smoke artifacts and placeholder code

### Validation Framework:
- Token distribution verified in datasets (36 vs 0 `<zwx>` occurrences)
- Consistent hyperparameters for fair comparison
- Deterministic seeds (11, 12, 13) for reproducible results
- File size validation (>30KB for real images)

## 📊 **Next Steps**

1. **Resolve ai-toolkit dependencies** - complete the diffusers/optimum-quanto compatibility
2. **Execute both training experiments** - zero-prior and baseline with identical settings
3. **Generate comparison predictions** - same prompts and seeds for both models
4. **Validate results** - ensure real weights and images, no fake outputs
5. **Document findings** - quantitative comparison of zero-prior vs baseline performance

---

### Status: **Code Ready, Dependencies In Progress**
The implementation is complete and fail-safe. All experiments will run automatically once the ai-toolkit dependency chain resolves.