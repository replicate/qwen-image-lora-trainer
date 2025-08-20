
# Qwen Image LoRA

[![Replicate](https://replicate.com/qwen/qwen-image-lora/badge)](https://replicate.com/qwen/qwen-image-lora)

Fine-tunable Qwen Image model with exceptional composition abilities. Train custom LoRAs for any style or subject.

## Training

Train your own LoRA on [Replicate](https://replicate.com/qwen/qwen-image-lora/train) or locally:

```bash
cog train -i dataset=@your-images.zip -i default_caption="A photo of a person named <>"
```

Training runs on Nvidia H100 GPU hardware and outputs a ZIP file with your LoRA weights.

## Inference

Generate images using your trained LoRA:

```bash
cog predict -i prompt="A beautiful sunset" -i replicate_weights=@your-trained-lora.zip
```

## Local Development

```bash
git clone --recursive https://github.com/your-repo/qwen-image-lora-trainer.git
cd qwen-image-lora-trainer
```

Then use `cog train` and `cog predict` as shown above.

## Dataset Format

Your training ZIP should contain images (`.jpg`, `.png`, `.webp`) and optionally matching `.txt` caption files:

```
dataset.zip
├── photo1.jpg
├── photo1.txt        # "A photo of a person named <>"
├── photo2.jpg
└── photo3.jpg        # Will use default_caption
```

## Important: Qwen Prompting

**Critical**: Qwen is extremely sensitive to prompting and differs from other image models. Do NOT use abstract tokens like "TOK", "sks", or meaningless identifiers. 

Instead, use descriptive, familiar words that closely match your actual images:
- ✅ "person", "man", "woman", "dog", "cat", "building", "car"  
- ❌ "TOK", "sks", "subj", random tokens

Every token carries meaning - the model learns by overriding specific descriptive concepts rather than learning new tokens. Be precise and descriptive about what's actually in your images.

## Notes

- Training typically takes 15-30 minutes depending on dataset size
- Runs on Nvidia H100 GPU hardware on Replicate
