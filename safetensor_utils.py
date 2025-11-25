"""Utilities for converting LoRA safetensors to match Pruna's expected format."""

import os
import torch
from typing import Dict, Any, Optional
from safetensors import safe_open
from safetensors.torch import save_file as save_file_torch
import hashlib
from pathlib import Path


class RenameError(Exception):
    """Custom exception for renaming errors."""
    def __init__(self, message: str, code: int = 1):
        super().__init__(message)
        self.code = code


def tensor_checksum_pt(tensor: torch.Tensor) -> str:
    """Compute a checksum for a PyTorch tensor."""
    tensor_np = tensor.detach().cpu().numpy()
    return hashlib.md5(tensor_np.tobytes()).hexdigest()


def rename_key(key: str) -> str:
    """
    Rename a single key from 'diffusion_model' to 'transformer'.
    
    Args:
        key: Original key name
        
    Returns:
        Renamed key with 'transformer' prefix if it had 'diffusion_model', 
        otherwise returns the original key unchanged.
    """
    if key.startswith("diffusion_model."):
        return key.replace("diffusion_model.", "transformer.", 1)
    return key


def rename_lora_keys_for_pruna(
    src_path: str,
    out_path: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Rename LoRA keys from "diffusion_model" -> "transformer" in a .safetensors file.
    This ensures compatibility with Pruna's expected format.
    
    Args:
        src_path: Path to input .safetensors file
        out_path: Optional output path (defaults to overwriting the input)
        dry_run: If True, only shows what would be renamed without writing
        
    Returns:
        Dictionary with conversion summary
        
    Raises:
        RenameError: If there are issues with the conversion
    """
    in_path = os.path.abspath(src_path)
    
    if not os.path.exists(in_path):
        raise RenameError(f"Input file not found: {in_path}", code=1)
    
    # Default to overwriting the input file if no output specified
    if out_path is None:
        out_path = in_path
    else:
        out_path = os.path.abspath(out_path)
    
    print(f"Reading: {in_path}")
    
    # Read the safetensors file
    with safe_open(in_path, framework="pt") as f:
        orig_keys = list(f.keys())
        
        if dry_run:
            print("Planned key changes:")
            planned_changed = 0
            for k in orig_keys:
                nk = rename_key(k)
                if nk != k:
                    planned_changed += 1
                    print(f"  {k}  ->  {nk}")
                else:
                    print(f"  {k}  (unchanged)")
            print("Dry run complete.")
            return {
                "input_path": in_path,
                "output_path": None,
                "num_tensors": len(orig_keys),
                "num_renamed": planned_changed,
                "dry_run": True,
            }
        
        # Check if any keys need renaming
        needs_rename = any(k.startswith("diffusion_model.") for k in orig_keys)
        
        if not needs_rename:
            print("No keys need renaming (already in correct format)")
            return {
                "input_path": in_path,
                "output_path": out_path,
                "num_tensors": len(orig_keys),
                "num_renamed": 0,
                "dry_run": False,
            }
        
        # Perform the actual renaming
        renamed_tensors: Dict[str, torch.Tensor] = {}
        meta: Dict[str, Dict[str, Any]] = {}
        
        for k in orig_keys:
            t = f.get_tensor(k)  # torch.Tensor (lazy loaded)
            meta[k] = {
                "shape": tuple(t.shape),
                "dtype": str(t.dtype),
                "checksum": tensor_checksum_pt(t),
            }
            nk = rename_key(k)
            
            if nk in renamed_tensors:
                raise RenameError(
                    f"ERROR: Collision after renaming: '{nk}' already exists",
                    code=2,
                )
            renamed_tensors[nk] = t
    
    print(f"Writing: {out_path}")
    save_file_torch(renamed_tensors, out_path)
    
    # Verify the conversion
    print("Verifying...")
    with safe_open(out_path, framework="pt") as g:
        new_keys = list(g.keys())
        
        if len(new_keys) != len(meta):
            raise RenameError(
                f"ERROR: Tensor count mismatch: {len(new_keys)} vs {len(meta)}",
                code=3,
            )
        
        reverse_map = {rename_key(k): k for k in meta.keys()}
        
        for nk in new_keys:
            if nk not in reverse_map:
                raise RenameError(f"ERROR: Unexpected key after rename: {nk}", code=4)
            
            ok = reverse_map[nk]
            t_new = g.get_tensor(nk)
            m = meta[ok]
            
            if tuple(t_new.shape) != m["shape"] or str(t_new.dtype) != m["dtype"]:
                raise RenameError(
                    (
                        f"ERROR: Mismatch for {nk}: shape/dtype changed\n"
                        f"  expected {m['shape']} {m['dtype']} got {tuple(t_new.shape)} {t_new.dtype}"
                    ),
                    code=5,
                )
            
            if tensor_checksum_pt(t_new) != m["checksum"]:
                raise RenameError(f"ERROR: Content changed for {nk}", code=6)
    
    changed = sum(1 for k in meta if rename_key(k) != k)
    print("Success ✅")
    print(f"  Input tensors : {len(meta)}")
    print(f"  Renamed keys  : {changed}")
    print(f"  Output file   : {out_path}")
    
    return {
        "input_path": in_path,
        "output_path": out_path,
        "num_tensors": len(meta),
        "num_renamed": changed,
        "dry_run": False,
    }
