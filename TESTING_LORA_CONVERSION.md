# Reproducible Experiment: LoRA Safetensor Key Conversion and Inference Equivalence

This document is a complete, step-by-step recipe to reproduce our experiment verifying that converting LoRA safetensor keys from `diffusion_model.*` to `transformer.*` preserves model behavior. It is written so an LLM or a human can understand the intent, the file formats, the commands, and the expected outcomes without any extra context.

The experiment runs 3 predictions with the exact same settings except for the LoRA weights:
1) Base model (no LoRA)
2) Original LoRA (as downloaded)
3) Converted LoRA (keys renamed for Pruna/Qwen compatibility)

Expected outcome:
- Base model image differs from the LoRA images
- Original and Converted LoRA images are bitwise-identical

Prompt used: "A photo of a person named Sakib"

---

## Definitions (explicit context)

- LoRA: Low-Rank Adaptation weights trained to adapt a base diffusion model. In this repo, LoRA weights are loaded on top of the Qwen Image base model.
- Safetensors: A tensor serialization format used for the LoRA weights file (e.g., `lora.safetensors`).
- Conversion: Renaming safetensor keys from the source format used by training (`diffusion_model.*`) to the format expected by Pruna/Qwen (`transformer.*`). Only keys are renamed; tensor data are unchanged.
- Zip containing safetensors: A `.zip` file whose top-level contains a single file named `lora.safetensors`. This is the format expected by `-i replicate_weights=@...` in `cog predict`.
- LoRA strength: The scalar applied to the LoRA during inference. In this repo it is `lora_scale`. The default is `1.0`. Using the same strength across runs is essential to a fair comparison.
- Experiment (in this document): The exact 3-run procedure with identical parameters (prompt, seed, steps, guidance, resolution, LoRA strength), changing only the LoRA condition (none vs original vs converted), saving each result with a distinct filename, and verifying equivalence/difference via checksums.

---

## Requirements

- Linux with NVIDIA GPU and CUDA
- Docker configured with GPU access
- Cog CLI (we used `cog==0.16.2`): `pip install cog==0.16.2`
- Python 3.10+ with `safetensors` and `torch`: `pip install safetensors torch`
- This repository checked out locally and able to run `cog predict`

Repo root referenced below: `/home/ubuntu/qwen-image-lora-trainer`

---

## Fixed parameters (to guarantee identical settings)

We use the same prompt, seed, LoRA scale, steps, guidance, and resolution across all runs.

```bash
# From the repo root
cd /home/ubuntu/qwen-image-lora-trainer

# Constants for the experiment
P="A photo of a person named Sakib"
SEED=42
LORA_SCALE=1.0        # LoRA strength (keep identical for all LoRA runs)
STEPS=20
GUIDANCE=4
WIDTH=1024
HEIGHT=1024

# Output folder for images and logs
mkdir -p real_comparison_test
```

Note on resolution inputs:
- If your `predict.py` uses Pruna-style inputs, you can use `-i aspect_ratio="1:1" -i image_size="optimize_for_speed"`.
- If you encounter validation bugs, pass `-i width=$WIDTH -i height=$HEIGHT` explicitly (shown in the commands below).

---

## 0) Optional: Train a LoRA on H100/H200 and use the produced zip

If you're on an NVIDIA H100/H200 box and want to validate the packaging-time conversion in `train.py` itself, you can train a small LoRA and use the resulting zip for inference.

```bash
# From the repo root
cd /home/ubuntu/qwen-image-lora-trainer

# Option A (recommended): download dataset then pass as file
wget -O me-dataset.zip "https://replicate.delivery/pbxt/NYTtOHyAWc091ZOVdLaWqrsZ5bxOoFBxasQIhHa9ACf0VULb/me-dataset.zip"
cog train \
  -i dataset=@me-dataset.zip \
  -i default_caption="A photo of a person named Sakib"

# Option B (if passing URL works in your environment)
# cog train -i dataset="https://replicate.delivery/pbxt/NYTtOHyAWc091ZOVdLaWqrsZ5bxOoFBxasQIhHa9ACf0VULb/me-dataset.zip" \
#          -i default_caption="A photo of a person named Sakib"

# The training job writes a zip to /tmp named like /tmp/qwen_lora_<timestamp>_trained.zip
LATEST=$(ls -t /tmp/*_trained.zip | head -n1)
echo "Using: $LATEST"

# Prepare folder and unpack
mkdir -p real_lora_test
cp "$LATEST" real_lora_test/trained_lora.zip
unzip -q -o real_lora_test/trained_lora.zip -d real_lora_test
ls -la real_lora_test

# Sanity check keys (expect 'transformer' prefix because train.py converts at packaging)
python3 - << 'PY'
from safetensors import safe_open
st = safe_open('real_lora_test/lora.safetensors', framework='pt')
keys = list(st.keys())
print('Unique prefixes:', {k.split('.')[0] for k in keys})
PY

# Then continue at step 3 below to run predictions using this new zip
# (or you can still run the original vs converted comparison in steps 1-2).
```

Notes:
- This primarily validates the conversion inside `train.py#create_output_archive`.
- For an "original vs converted" equivalence test, use the download-based flow in steps 1-2.

---

## 1) Download a real trained LoRA and inspect its keys

We use a real LoRA delivered as a zip from Replicate. It unpacks to `lora.safetensors` (and config files).

```bash
mkdir -p real_lora_test
cd real_lora_test
wget -O trained_lora.zip "https://replicate.delivery/xezq/fmVO5L9GNuXPZqGRX7DxFH1TEr1NHk197GIwXiQkaaY4SdmKA/qwen_lora_1755705850_trained.zip"
unzip -q trained_lora.zip
ls -la
# Expect: lora.safetensors, config.yaml, settings.txt
```

Confirm keys use the training prefix `diffusion_model.`:

```bash
python3 - << 'PY'
from safetensors import safe_open
st = safe_open('real_lora_test/lora.safetensors', framework='pt')
keys = list(st.keys())
print('Total keys:', len(keys))
print('First 5 keys:')
for k in sorted(keys)[:5]:
    print(' ', k)
print('Unique prefixes:', {k.split('.')[0] for k in keys})
PY
```

---

## 2) Convert the safetensors keys (diffusion_model.* -> transformer.*)

Use the repository utility to test the same logic used by training/packaging:

```bash
python3 - << 'PY'
from safetensor_utils import rename_lora_keys_for_pruna
res = rename_lora_keys_for_pruna(
    src_path='real_lora_test/lora.safetensors',
    out_path='real_lora_test/lora_converted.safetensors',
    dry_run=False,
)
print(res)
PY
```

Package both versions as zip files expected by `cog predict` (must contain a top-level file named `lora.safetensors`):

```bash
# Original zip (top-level lora.safetensors)
cd real_lora_test
zip -q original_lora_real.zip lora.safetensors

# Converted zip (rename inside the archive to lora.safetensors)
mkdir -p /tmp/converted_zip_staging
cp lora_converted.safetensors /tmp/converted_zip_staging/lora.safetensors
(cd /tmp/converted_zip_staging && zip -q /home/ubuntu/qwen-image-lora-trainer/real_lora_test/converted_lora_real.zip lora.safetensors)
cd ..
```

What "Zip containing safetensors" means in this experiment:
- The zip passed to `-i replicate_weights=@...` must contain a single file named `lora.safetensors` at the top level, e.g.:
```
original_lora_real.zip
└── lora.safetensors
```

---

## 3) Run the 3 predictions (same settings for all runs)

We run three `cog predict` commands. Each writes a file named `output.png`. Immediately after each prediction, we rename the file to avoid it being overwritten by the next run.

Base model (no LoRA):

```bash
cog predict \
  -i prompt="$P" \
  -i width=$WIDTH -i height=$HEIGHT \
  -i seed=$SEED -i num_inference_steps=$STEPS -i guidance=$GUIDANCE \
  -i output_format="png" -i go_fast=false -i enhance_prompt=false
# Rename the output immediately so it’s not overwritten by the next run
mv -f output.png real_comparison_test/test1_base_model.png
```

Original LoRA (diffusion_model.* keys):

```bash
cog predict \
  -i prompt="$P" \
  -i replicate_weights=@real_lora_test/original_lora_real.zip \
  -i width=$WIDTH -i height=$HEIGHT \
  -i seed=$SEED -i num_inference_steps=$STEPS -i guidance=$GUIDANCE \
  -i output_format="png" -i lora_scale=$LORA_SCALE \
  -i go_fast=false -i enhance_prompt=false
mv -f output.png real_comparison_test/test2_original_lora.png
```

Converted LoRA (transformer.* keys):

```bash
cog predict \
  -i prompt="$P" \
  -i replicate_weights=@real_lora_test/converted_lora_real.zip \
  -i width=$WIDTH -i height=$HEIGHT \
  -i seed=$SEED -i num_inference_steps=$STEPS -i guidance=$GUIDANCE \
  -i output_format="png" -i lora_scale=$LORA_SCALE \
  -i go_fast=false -i enhance_prompt=false
mv -f output.png real_comparison_test/test3_converted_lora.png
```

Notes:
- "Same everything" means the prompt, seed, number of steps, guidance scale, resolution, and LoRA strength (`lora_scale`) are identical across all runs. The only change is whether `replicate_weights` is provided, and which zip file is used.
- Look for `LoRA loaded: dim=..., alpha=..., scale=...` in the logs for the LoRA runs.

---

## 4) Verify the results

The base image should differ from the LoRA images. The LoRA images (original vs converted) should be exactly identical.

```bash
cd real_comparison_test
md5sum *.png
# Example from our run:
# 25f12cf54f6db31db4b64330e9cb042f  test1_base_model.png
# 9592222859080d9cc09c129e61f0ffb6  test2_original_lora.png
# 9592222859080d9cc09c129e61f0ffb6  test3_converted_lora.png
cd ..
```

Optional: quick size sanity check

```bash
ls -lh real_comparison_test/*.png
```

Optional: pixel diff (requires ImageMagick)

```bash
# Compares original vs converted; expect zero-difference
compare -metric AE real_comparison_test/test2_original_lora.png \
                 real_comparison_test/test3_converted_lora.png \
                 null:
```

---

## 5) Interpreting logs and common warnings

- You should see `Generation took X seconds` and `Total safe images: Y/Z` after each run.
- On LoRA runs, logs should include `LoRA loaded: dim=..., alpha=..., scale=...`.
- You may see `Missing keys` warnings if the loader expects a different key tokenization style. This does not affect the equivalence of original vs converted and is unrelated to the correctness of the key-prefix conversion.

---

## 6) Troubleshooting

- All three images are identical:
  - Ensure you are using a real trained LoRA (not dummy random weights).
  - Verify `lora_scale` is positive (e.g., `1.0`) and not `0`.
  - Confirm the zip structure contains a top-level `lora.safetensors`.
  - Check logs for `LoRA loaded: ...`. If absent, the LoRA may not be loading.
- Validation errors for width/height:
  - Switch between explicit `width/height` and `aspect_ratio + image_size` depending on your `predict.py` version.
- CUDA OOM or GPU busy:
  - Close other GPU jobs and retry. Cog will report GPU memory errors if resources are insufficient.

---

## 7) Optional cleanup

To remove test artifacts while keeping this documentation:

```bash
rm -rf real_lora_test real_comparison_test
```

---

## Conclusion

This experiment confirms that renaming safetensor keys from `diffusion_model.*` to `transformer.*` is a safe, behavior-preserving conversion for this repository:
- Base model (no LoRA) vs LoRA runs -> images differ (LoRA has an effect)
- Original vs Converted LoRA -> images are identical (conversion preserves tensors)

With this document alone, you can recreate the entire experiment end-to-end.
