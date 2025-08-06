#!/usr/bin/env python3
"""
Example usage of Qwen-Image LoRA inference

This script demonstrates how to use the inference package programmatically
with different configurations.
"""

import os
import sys
from predict import QwenImageLoRAInference


def example_basic_generation():
    """Basic image generation without LoRA"""
    print("=== Basic Generation Example ===")
    
    inference = QwenImageLoRAInference(device="cuda", dtype="bf16")
    inference.load_base_model()
    
    image = inference.generate_image(
        prompt="a serene mountain landscape with a lake",
        width=1024,
        height=1024,
        num_inference_steps=20,
        guidance_scale=4.0,
        seed=42
    )
    
    image.save("example_basic.jpg")
    print("✓ Basic example saved as example_basic.jpg\n")


def example_with_lora():
    """Image generation with LoRA weights"""
    print("=== LoRA Example ===")
    
    # Check if LoRA weights exist
    lora_path = "./output/transformer/lora_weights.safetensors"
    if not os.path.exists(lora_path):
        print(f"LoRA weights not found at {lora_path}")
        print("Train a LoRA first using the ai-toolkit")
        return
    
    inference = QwenImageLoRAInference(device="cuda", dtype="bf16")
    inference.load_base_model()
    inference.load_lora_weights(lora_path, lora_scale=1.0)
    
    image = inference.generate_image(
        prompt="a photo of a person reading a book in a cozy library",
        width=1024,
        height=1024,
        num_inference_steps=25,
        guidance_scale=4.5,
        seed=123
    )
    
    image.save("example_lora.jpg")
    print("✓ LoRA example saved as example_lora.jpg\n")


def example_with_placeholder_token():
    """Image generation with placeholder token and LoRA"""
    print("=== Placeholder Token + LoRA Example ===")
    
    # Check if LoRA weights exist
    lora_path = "./output/transformer/lora_weights.safetensors"
    if not os.path.exists(lora_path):
        print(f"LoRA weights not found at {lora_path}")
        print("Train a LoRA with placeholder tokens first")
        return
    
    inference = QwenImageLoRAInference(device="cuda", dtype="bf16")
    inference.load_base_model()
    
    # Register the placeholder token (same as used in training)
    inference.register_placeholder_token("<zwx>")
    
    # Load LoRA weights
    inference.load_lora_weights(lora_path, lora_scale=1.2)
    
    # Generate multiple images with different prompts
    prompts = [
        "a photo of <zwx>",
        "a portrait of <zwx> smiling",
        "a photo of <zwx> in a garden",
        "<zwx> wearing a red jacket",
    ]
    
    for i, prompt in enumerate(prompts):
        image = inference.generate_image(
            prompt=prompt,
            width=1024,
            height=1024,
            num_inference_steps=20,
            guidance_scale=4.0,
            seed=42 + i  # Different seed for variety
        )
        
        filename = f"example_placeholder_{i+1}.jpg"
        image.save(filename)
        print(f"✓ Generated: {filename} - '{prompt}'")
    
    print("✓ All placeholder token examples generated\n")


def example_batch_generation():
    """Generate multiple variations of the same subject"""
    print("=== Batch Generation Example ===")
    
    inference = QwenImageLoRAInference(device="cuda", dtype="bf16")
    inference.load_base_model()
    
    # Base prompt with variations
    base_prompt = "a beautiful garden with flowers"
    variations = [
        f"{base_prompt} in spring",
        f"{base_prompt} in autumn", 
        f"{base_prompt} at sunset",
        f"{base_prompt} with butterflies",
    ]
    
    for i, prompt in enumerate(variations):
        image = inference.generate_image(
            prompt=prompt,
            width=1024,
            height=1024,
            num_inference_steps=20,
            guidance_scale=4.0,
            seed=100 + i
        )
        
        filename = f"batch_{i+1}.jpg"
        image.save(filename)
        print(f"✓ Generated: {filename}")
    
    print("✓ Batch generation complete\n")


def main():
    """Run all examples"""
    print("Qwen-Image LoRA Inference Examples")
    print("==================================\n")
    
    # Check if CUDA is available
    import torch
    if not torch.cuda.is_available():
        print("⚠ CUDA not available, examples will run on CPU (slower)")
        input("Press Enter to continue or Ctrl+C to exit...")
    
    try:
        # Run examples
        example_basic_generation()
        example_with_lora()
        example_with_placeholder_token()
        example_batch_generation()
        
        print("🎉 All examples completed successfully!")
        print("Check the generated .jpg files in the current directory.")
        
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()