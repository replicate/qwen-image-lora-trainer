
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

