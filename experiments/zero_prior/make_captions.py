#!/usr/bin/env python3
"""
Automated caption generation for zero-prior placeholder token experiments.
Creates deterministic captions using consistent templates and subjects.
"""

import os
import sys
import json
import zipfile
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image
import random

# Deterministic subject templates for consistent experiments
CAPTION_TEMPLATES = [
    "a photo of {subject}",
    "a portrait of {subject}", 
    "a close-up of {subject}",
    "{subject} in natural lighting",
    "a professional photo of {subject}",
    "{subject} looking at camera",
    "a candid shot of {subject}",
    "{subject} smiling",
    "an image of {subject}",
    "{subject} in casual setting"
]

BASELINE_SUBJECTS = [
    "a person",
    "the subject", 
    "an individual",
    "someone",
    "the person in the photo"
]

def extract_dataset(zip_path: str, output_dir: str) -> List[str]:
    """Extract dataset and return list of image files"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_files = []
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for file_info in zf.infolist():
            if file_info.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                # Extract to output directory
                zf.extract(file_info, output_path)
                image_files.append(str(output_path / file_info.filename))
    
    return sorted(image_files)

def generate_baseline_captions(image_files: List[str], seed: int = 42) -> Dict[str, str]:
    """Generate baseline captions without placeholder tokens"""
    random.seed(seed)
    captions = {}
    
    for img_path in image_files:
        # Use deterministic selection based on file path
        img_name = Path(img_path).name
        template_idx = abs(hash(img_name)) % len(CAPTION_TEMPLATES)
        subject_idx = abs(hash(img_name + "subject")) % len(BASELINE_SUBJECTS)
        
        template = CAPTION_TEMPLATES[template_idx]
        subject = BASELINE_SUBJECTS[subject_idx]
        
        caption = template.format(subject=subject)
        captions[img_path] = caption
    
    return captions

def generate_zeropri_captions(image_files: List[str], placeholder_token: str = "<zwx>", seed: int = 42) -> Dict[str, str]:
    """Generate zero-prior captions with placeholder tokens"""
    random.seed(seed)
    captions = {}
    
    for img_path in image_files:
        # Use same deterministic selection for fair comparison
        img_name = Path(img_path).name
        template_idx = abs(hash(img_name)) % len(CAPTION_TEMPLATES)
        
        template = CAPTION_TEMPLATES[template_idx]
        caption = template.format(subject=placeholder_token)
        captions[img_path] = caption
    
    return captions

def write_caption_files(captions: Dict[str, str], output_dir: str):
    """Write individual .txt caption files"""
    output_path = Path(output_dir)
    
    for img_path, caption in captions.items():
        img_name = Path(img_path).stem
        caption_file = output_path / f"{img_name}.txt"
        
        with open(caption_file, 'w', encoding='utf-8') as f:
            f.write(caption)

def create_dataset_zip(image_dir: str, output_zip: str):
    """Create dataset zip with images and caption files"""
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        image_path = Path(image_dir)
        
        for file_path in image_path.iterdir():
            if file_path.is_file():
                # Add to zip with relative path
                zf.write(file_path, file_path.name)

def analyze_captions(captions: Dict[str, str]) -> Dict:
    """Analyze caption statistics"""
    stats = {
        "total_captions": len(captions),
        "unique_captions": len(set(captions.values())),
        "avg_length": sum(len(c.split()) for c in captions.values()) / len(captions),
        "template_distribution": {},
        "sample_captions": list(captions.values())[:5]
    }
    
    # Count template usage
    for caption in captions.values():
        template_key = caption.split(" of ")[0] + " of X" if " of " in caption else caption
        stats["template_distribution"][template_key] = stats["template_distribution"].get(template_key, 0) + 1
    
    return stats

def main():
    if len(sys.argv) < 2:
        print("Usage: python make_captions.py <dataset.zip> [placeholder_token]")
        print("  Creates baseline and zero-prior caption datasets")
        sys.exit(1)
    
    dataset_zip = sys.argv[1]
    placeholder_token = sys.argv[2] if len(sys.argv) > 2 else "<zwx>"
    
    if not Path(dataset_zip).exists():
        print(f"Error: Dataset {dataset_zip} not found")
        sys.exit(1)
    
    print(f"🏷️  Generating captions for {dataset_zip}")
    print(f"📝 Placeholder token: {placeholder_token}")
    
    # Create output directories
    base_dir = Path("experiments/zero_prior/datasets")
    baseline_dir = base_dir / "baseline"
    zeropri_dir = base_dir / "zeropri" 
    
    baseline_dir.mkdir(parents=True, exist_ok=True)
    zeropri_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract original dataset
    print("📦 Extracting dataset...")
    temp_dir = base_dir / "temp_extract"
    image_files = extract_dataset(dataset_zip, str(temp_dir))
    print(f"   Found {len(image_files)} images")
    
    # Generate baseline captions
    print("🏷️  Generating baseline captions...")
    baseline_captions = generate_baseline_captions(image_files)
    
    # Copy images to baseline directory
    for img_path in image_files:
        img_name = Path(img_path).name
        baseline_img = baseline_dir / img_name
        Image.open(img_path).save(baseline_img)
    
    write_caption_files(baseline_captions, str(baseline_dir))
    create_dataset_zip(str(baseline_dir), str(base_dir / "baseline_dataset.zip"))
    
    # Generate zero-prior captions  
    print("🔮 Generating zero-prior captions...")
    zeropri_captions = generate_zeropri_captions(image_files, placeholder_token)
    
    # Copy images to zero-prior directory
    for img_path in image_files:
        img_name = Path(img_path).name
        zeropri_img = zeropri_dir / img_name
        Image.open(img_path).save(zeropri_img)
    
    write_caption_files(zeropri_captions, str(zeropri_dir))
    create_dataset_zip(str(zeropri_dir), str(base_dir / "zeropri_dataset.zip"))
    
    # Generate analysis report
    print("📊 Analyzing captions...")
    baseline_stats = analyze_captions(baseline_captions)
    zeropri_stats = analyze_captions(zeropri_captions)
    
    analysis = {
        "dataset": str(Path(dataset_zip).name),
        "placeholder_token": placeholder_token,
        "baseline_stats": baseline_stats,
        "zeropri_stats": zeropri_stats,
        "sample_comparisons": []
    }
    
    # Create sample comparisons
    for img_path in list(image_files)[:5]:
        comparison = {
            "image": Path(img_path).name,
            "baseline_caption": baseline_captions[img_path],
            "zeropri_caption": zeropri_captions[img_path]
        }
        analysis["sample_comparisons"].append(comparison)
    
    # Save analysis
    with open(base_dir / "caption_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Print summary
    print(f"\n✅ Caption generation complete!")
    print(f"📄 Baseline captions: {baseline_stats['total_captions']}")
    print(f"📄 Zero-prior captions: {zeropri_stats['total_captions']}")
    print(f"📊 Average length: {baseline_stats['avg_length']:.1f} words")
    print(f"\n📁 Output files:")
    print(f"   - {base_dir}/baseline_dataset.zip")
    print(f"   - {base_dir}/zeropri_dataset.zip")
    print(f"   - {base_dir}/caption_analysis.json")
    
    print(f"\n🔍 Sample comparisons:")
    for comp in analysis["sample_comparisons"][:3]:
        print(f"   {comp['image']}:")
        print(f"     Baseline: {comp['baseline_caption']}")
        print(f"     Zero-prior: {comp['zeropri_caption']}")
    
    # Clean up temp directory
    import shutil
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()