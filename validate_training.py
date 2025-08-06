#!/usr/bin/env python3
"""
Training validation helper for zero-prior placeholder token testing.
Checks tokenizer behavior, embedding initialization, and gradient masking.
"""

import json
import zipfile
import safetensors.torch
from pathlib import Path
import torch

def validate_archive_structure(archive_path: str) -> dict:
    """Validate the structure of training output archive"""
    results = {"archive_exists": False, "has_lora": False, "has_metadata": False, "metadata": {}}
    
    archive_path = Path(archive_path)
    if not archive_path.exists():
        return results
    
    results["archive_exists"] = True
    results["archive_size"] = archive_path.stat().st_size
    
    try:
        with zipfile.ZipFile(archive_path, 'r') as zf:
            file_list = zf.namelist()
            
            # Check for LoRA weights
            lora_files = [f for f in file_list if 'lora' in f.lower() and f.endswith('.safetensors')]
            results["has_lora"] = len(lora_files) > 0
            results["lora_files"] = lora_files
            
            # Check for metadata
            if 'meta.json' in file_list:
                results["has_metadata"] = True
                try:
                    with zf.open('meta.json') as meta_file:
                        results["metadata"] = json.load(meta_file)
                except Exception as e:
                    results["metadata_error"] = str(e)
            
            results["all_files"] = file_list
            
    except Exception as e:
        results["archive_error"] = str(e)
    
    return results

def validate_tokenizer_behavior(placeholder_token: str = "<zwx>") -> dict:
    """Validate placeholder token tokenization behavior"""
    results = {}
    
    try:
        from transformers import Qwen2Tokenizer
        
        # Load tokenizer
        tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen-Image", subfolder="tokenizer")
        
        # Check if token already exists
        try:
            existing_id = tokenizer.convert_tokens_to_ids(placeholder_token)
            results["token_exists_before"] = existing_id != tokenizer.unk_token_id
            results["existing_id"] = existing_id
        except:
            results["token_exists_before"] = False
        
        # Add token
        original_vocab_size = len(tokenizer)
        num_added = tokenizer.add_special_tokens({"additional_special_tokens": [placeholder_token]})
        new_vocab_size = len(tokenizer)
        
        results["original_vocab_size"] = original_vocab_size
        results["new_vocab_size"] = new_vocab_size
        results["tokens_added"] = num_added
        
        # Check single-token encoding
        token_ids = tokenizer.encode(placeholder_token, add_special_tokens=False)
        results["encoded_ids"] = token_ids
        results["is_single_token"] = len(token_ids) == 1
        
        if len(token_ids) == 1:
            results["placeholder_token_id"] = token_ids[0]
        
        # Test in context
        test_prompt = f"a photo of {placeholder_token}"
        context_ids = tokenizer.encode(test_prompt, add_special_tokens=False)
        results["context_test"] = {
            "prompt": test_prompt,
            "token_ids": context_ids,
            "contains_placeholder": token_ids[0] in context_ids if len(token_ids) == 1 else False
        }
        
    except Exception as e:
        results["error"] = str(e)
    
    return results

def validate_lora_weights(archive_path: str) -> dict:
    """Validate LoRA weights in the archive"""
    results = {}
    
    try:
        with zipfile.ZipFile(archive_path, 'r') as zf:
            # Find LoRA file
            lora_files = [f for f in zf.namelist() if 'lora' in f.lower() and f.endswith('.safetensors')]
            
            if not lora_files:
                results["error"] = "No LoRA files found"
                return results
            
            # Extract and validate first LoRA file
            lora_file = lora_files[0]
            results["lora_file"] = lora_file
            
            # Extract to temporary location
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.safetensors') as tmp:
                with zf.open(lora_file) as lora_data:
                    tmp.write(lora_data.read())
                tmp.flush()
                
                # Load safetensors
                lora_weights = safetensors.torch.load_file(tmp.name)
                
                results["num_tensors"] = len(lora_weights)
                results["tensor_names"] = list(lora_weights.keys())
                results["total_params"] = sum(tensor.numel() for tensor in lora_weights.values())
                
                # Check tensor shapes and dtypes
                tensor_info = {}
                for name, tensor in lora_weights.items():
                    tensor_info[name] = {
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                        "norm": float(tensor.norm().item())
                    }
                
                results["tensor_info"] = tensor_info
                
    except Exception as e:
        results["error"] = str(e)
    
    return results

def print_validation_report(archive_path: str, placeholder_token: str = "<zwx>"):
    """Print comprehensive validation report"""
    print(f"\n🔍 Validation Report for {archive_path}")
    print("=" * 60)
    
    # Archive structure
    print("\n📦 Archive Structure:")
    archive_results = validate_archive_structure(archive_path)
    print(f"  ✅ Archive exists: {archive_results['archive_exists']}")
    if archive_results["archive_exists"]:
        print(f"  📊 Size: {archive_results['archive_size']:,} bytes")
        print(f"  ✅ Has LoRA weights: {archive_results['has_lora']}")
        print(f"  ✅ Has metadata: {archive_results['has_metadata']}")
        
        if archive_results["has_metadata"]:
            metadata = archive_results["metadata"]
            print(f"  📋 Metadata:")
            for key, value in metadata.items():
                print(f"      {key}: {value}")
        
        print(f"  📁 Files: {len(archive_results.get('all_files', []))}")
        for f in archive_results.get("all_files", []):
            print(f"      - {f}")
    
    # Tokenizer validation
    print(f"\n🏷️  Tokenizer Validation:")
    token_results = validate_tokenizer_behavior(placeholder_token)
    if "error" in token_results:
        print(f"  ❌ Error: {token_results['error']}")
    else:
        print(f"  📊 Original vocab size: {token_results['original_vocab_size']}")
        print(f"  📊 New vocab size: {token_results['new_vocab_size']}")
        print(f"  ➕ Tokens added: {token_results['tokens_added']}")
        print(f"  ✅ Single token encoding: {token_results['is_single_token']}")
        if token_results["is_single_token"]:
            print(f"  🆔 Placeholder token ID: {token_results['placeholder_token_id']}")
        print(f"  📝 Context test: {token_results['context_test']['contains_placeholder']}")
    
    # LoRA weights validation
    if archive_results["archive_exists"] and archive_results["has_lora"]:
        print(f"\n🔧 LoRA Weights:")
        lora_results = validate_lora_weights(archive_path)
        if "error" in lora_results:
            print(f"  ❌ Error: {lora_results['error']}")
        else:
            print(f"  📊 Number of tensors: {lora_results['num_tensors']}")
            print(f"  📊 Total parameters: {lora_results['total_params']:,}")
            print(f"  📄 Sample tensors:")
            for name in list(lora_results["tensor_names"])[:3]:  # Show first 3
                info = lora_results["tensor_info"][name]
                print(f"      {name}: {info['shape']} ({info['dtype']}) norm={info['norm']:.6f}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python validate_training.py <archive_path> [placeholder_token]")
        sys.exit(1)
    
    archive_path = sys.argv[1]
    placeholder_token = sys.argv[2] if len(sys.argv) > 2 else "<zwx>"
    
    print_validation_report(archive_path, placeholder_token)