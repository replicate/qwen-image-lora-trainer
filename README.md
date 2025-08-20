
## setup
- `git submodule update --init --recursive`
- `pip install -r ai-toolkit/requirements.txt` (maybe in a venv)
- `unzip zeke.zip`, delete all .txt files in here if any exist

## To launch training
- adjust `dataset.folder_path` in `qwen_config.yaml` to point to the zeke dataset
- cd to `ai-toolkit` and run `python run.py ../qwen_config.yaml`

## To run inference with trained LoRA weights
- adjust `LORA_FILE_PATH` in `qwen_lora_inference.py` to point to the LoRA weights
- run `python qwen_lora_inference.py`

## Cog inference (Replicate-style)
```bash
cog predict -i prompt="A beautiful sunset" -i aspect_ratio="16:9" -i image_size="optimize_for_quality"
```

Aspect ratios: `1:1`, `16:9`, `9:16`, `4:3`, `3:4`, `3:2`, `2:3`  
Image sizes: `optimize_for_quality` (~1.5MP), `optimize_for_speed` (~0.8MP)
