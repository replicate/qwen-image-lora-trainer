#!/usr/bin/env python3
"""
Generate side-by-side visual comparisons for zero-prior experiments.
Creates comparison grids and analysis visualizations.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import json

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️  PIL not available - generating text-based comparisons")

try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  Matplotlib not available - skipping charts")

def create_side_by_side_comparison(baseline_img: str, zeropri_img: str, 
                                  baseline_prompt: str, zeropri_prompt: str,
                                  output_path: str, title: str = "Comparison"):
    """Create side-by-side image comparison"""
    if not PIL_AVAILABLE:
        return create_text_comparison(baseline_img, zeropri_img, baseline_prompt, zeropri_prompt, output_path)
    
    try:
        # Load images
        img1 = Image.open(baseline_img).convert('RGB')
        img2 = Image.open(zeropri_img).convert('RGB')
        
        # Resize to common dimensions
        target_size = (512, 512)
        img1 = img1.resize(target_size, Image.Resampling.LANCZOS)
        img2 = img2.resize(target_size, Image.Resampling.LANCZOS)
        
        # Create comparison canvas
        margin = 20
        text_height = 80
        canvas_width = target_size[0] * 2 + margin * 3
        canvas_height = target_size[1] + margin * 2 + text_height
        
        canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
        
        # Paste images
        canvas.paste(img1, (margin, text_height + margin))
        canvas.paste(img2, (target_size[0] + margin * 2, text_height + margin))
        
        # Add text labels
        draw = ImageDraw.Draw(canvas)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()
            title_font = ImageFont.load_default()
        
        # Title
        title_width = draw.textlength(title, font=title_font)
        draw.text(((canvas_width - title_width) // 2, 10), title, fill='black', font=title_font)
        
        # Labels
        draw.text((margin, text_height - 30), "Baseline", fill='black', font=font)
        draw.text((target_size[0] + margin * 2, text_height - 30), "Zero-Prior", fill='black', font=font)
        
        # Prompts (truncated)
        max_prompt_len = 40
        baseline_short = baseline_prompt[:max_prompt_len] + "..." if len(baseline_prompt) > max_prompt_len else baseline_prompt
        zeropri_short = zeropri_prompt[:max_prompt_len] + "..." if len(zeropri_prompt) > max_prompt_len else zeropri_prompt
        
        draw.text((margin, text_height - 15), f'"{baseline_short}"', fill='gray', font=font)
        draw.text((target_size[0] + margin * 2, text_height - 15), f'"{zeropri_short}"', fill='gray', font=font)
        
        # Save comparison
        canvas.save(output_path)
        return True
        
    except Exception as e:
        print(f"❌ Failed to create visual comparison: {e}")
        return create_text_comparison(baseline_img, zeropri_img, baseline_prompt, zeropri_prompt, output_path)

def create_text_comparison(baseline_img: str, zeropri_img: str,
                          baseline_prompt: str, zeropri_prompt: str,
                          output_path: str) -> bool:
    """Create text-based comparison when PIL unavailable"""
    try:
        with open(output_path.replace('.png', '.txt'), 'w') as f:
            f.write("SIDE-BY-SIDE COMPARISON\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"BASELINE IMAGE: {baseline_img}\n")
            f.write(f"Prompt: {baseline_prompt}\n\n")
            f.write(f"ZERO-PRIOR IMAGE: {zeropri_img}\n")
            f.write(f"Prompt: {zeropri_prompt}\n\n")
        return True
    except Exception as e:
        print(f"❌ Failed to create text comparison: {e}")
        return False

def create_metrics_chart(metrics_data: Dict, output_path: str):
    """Create metrics comparison chart"""
    if not MATPLOTLIB_AVAILABLE:
        return create_text_metrics(metrics_data, output_path)
    
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Zero-Prior vs Baseline Metrics Comparison', fontsize=16)
        
        # CLIP similarity
        if 'baseline' in metrics_data and 'zeropri' in metrics_data:
            baseline_clip = metrics_data['baseline'].get('clip_similarity', {}).get('mean', 0)
            zeropri_clip = metrics_data['zeropri'].get('clip_similarity', {}).get('mean', 0)
            
            ax1.bar(['Baseline', 'Zero-Prior'], [baseline_clip, zeropri_clip], 
                   color=['lightblue', 'lightcoral'])
            ax1.set_title('CLIP Text-Image Similarity')
            ax1.set_ylabel('Similarity Score')
            ax1.set_ylim(0, 1)
            
            # Diversity
            baseline_div = metrics_data['baseline'].get('diversity', 0)
            zeropri_div = metrics_data['zeropri'].get('diversity', 0)
            
            ax2.bar(['Baseline', 'Zero-Prior'], [baseline_div, zeropri_div],
                   color=['lightblue', 'lightcoral'])
            ax2.set_title('Image Diversity (LPIPS)')
            ax2.set_ylabel('Diversity Score')
            
            # Quality metrics
            baseline_qual = metrics_data['baseline'].get('quality', {})
            zeropri_qual = metrics_data['zeropri'].get('quality', {})
            
            metrics = ['sharpness', 'contrast', 'brightness']
            baseline_vals = [baseline_qual.get(m, {}).get('mean', 0) for m in metrics]
            zeropri_vals = [zeropri_qual.get(m, {}).get('mean', 0) for m in metrics]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            ax3.bar(x - width/2, baseline_vals, width, label='Baseline', color='lightblue')
            ax3.bar(x + width/2, zeropri_vals, width, label='Zero-Prior', color='lightcoral')
            ax3.set_title('Perceptual Quality')
            ax3.set_ylabel('Quality Score')
            ax3.set_xticks(x)
            ax3.set_xticklabels(metrics)
            ax3.legend()
            
            # Summary comparison
            comparison = metrics_data.get('comparison', {})
            improvements = {
                'CLIP Similarity': comparison.get('clip_similarity_diff', 0),
                'Diversity Ratio': comparison.get('diversity_ratio', 1) - 1,
                'Sharpness': comparison.get('quality_comparison', {}).get('sharpness_diff', 0)
            }
            
            colors = ['green' if v > 0 else 'red' for v in improvements.values()]
            ax4.bar(improvements.keys(), improvements.values(), color=colors)
            ax4.set_title('Improvement Summary')
            ax4.set_ylabel('Change from Baseline')
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return True
        
    except Exception as e:
        print(f"❌ Failed to create metrics chart: {e}")
        return create_text_metrics(metrics_data, output_path)

def create_text_metrics(metrics_data: Dict, output_path: str) -> bool:
    """Create text-based metrics summary"""
    try:
        with open(output_path.replace('.png', '.txt'), 'w') as f:
            f.write("METRICS COMPARISON SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            if 'comparison' in metrics_data:
                comp = metrics_data['comparison']
                f.write("KEY FINDINGS:\n")
                f.write(f"- CLIP Similarity Change: {comp.get('clip_similarity_diff', 0):+.3f}\n")
                f.write(f"- Diversity Ratio: {comp.get('diversity_ratio', 1):.2f}x\n")
                f.write(f"- Quality Improvement: {comp.get('quality_comparison', {}).get('sharpness_diff', 0):+.3f}\n\n")
            
            for mode in ['baseline', 'zeropri']:
                if mode in metrics_data:
                    f.write(f"{mode.upper()} RESULTS:\n")
                    data = metrics_data[mode]
                    f.write(f"- CLIP Similarity: {data.get('clip_similarity', {}).get('mean', 0):.3f}\n")
                    f.write(f"- Diversity: {data.get('diversity', 0):.3f}\n")
                    f.write(f"- Sharpness: {data.get('quality', {}).get('sharpness', {}).get('mean', 0):.3f}\n")
                    f.write("\n")
        
        return True
    except Exception as e:
        print(f"❌ Failed to create text metrics: {e}")
        return False

def generate_comparison_grid(base_dir: str = "experiments/zero_prior"):
    """Generate comprehensive comparison visualizations"""
    base_path = Path(base_dir)
    outputs_dir = base_path / "outputs"
    comparison_dir = base_path / "comparisons"
    comparison_dir.mkdir(exist_ok=True)
    
    print("🖼️  Generating visual comparisons...")
    
    # Load configuration for prompts
    config_path = base_path / "configs" / "common.yaml"
    eval_prompts = {
        'baseline': ['a photo of a person', 'a portrait of the subject', 'a close-up of an individual'],
        'zeropri': ['a photo of <zwx>', 'a portrait of <zwx>', 'a close-up of <zwx>']
    }
    
    if config_path.exists():
        try:
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
            eval_prompts = config['experiment']['eval_prompts']
        except Exception as e:
            print(f"⚠️  Could not load config: {e}")
    
    # Find image pairs
    baseline_images = sorted(list(outputs_dir.glob("baseline_image_*.png")))
    zeropri_images = sorted(list(outputs_dir.glob("zeropri_image_*.png")))
    
    if not baseline_images or not zeropri_images:
        print("⚠️  No experiment images found - creating mock comparisons")
        
        # Create mock comparison text files
        for i in range(3):
            comparison_path = comparison_dir / f"comparison_{i+1:02d}.txt"
            with open(comparison_path, 'w') as f:
                f.write(f"MOCK COMPARISON {i+1}\n")
                f.write("=" * 30 + "\n\n")
                f.write(f"Baseline prompt: {eval_prompts['baseline'][i] if i < len(eval_prompts['baseline']) else 'a photo of a person'}\n")
                f.write(f"Zero-prior prompt: {eval_prompts['zeropri'][i] if i < len(eval_prompts['zeropri']) else 'a photo of <zwx>'}\n\n")
                f.write("Note: Actual experiment images would be displayed here\n")
        
        print(f"   Created {3} mock comparison files")
        return 3
    
    # Generate pairwise comparisons
    comparisons_created = 0
    max_comparisons = min(len(baseline_images), len(zeropri_images), len(eval_prompts['baseline']))
    
    for i in range(max_comparisons):
        baseline_img = str(baseline_images[i])
        zeropri_img = str(zeropri_images[i])
        baseline_prompt = eval_prompts['baseline'][i]
        zeropri_prompt = eval_prompts['zeropri'][i]
        
        comparison_path = comparison_dir / f"comparison_{i+1:02d}.png"
        title = f"Comparison {i+1}"
        
        success = create_side_by_side_comparison(
            baseline_img, zeropri_img, 
            baseline_prompt, zeropri_prompt,
            str(comparison_path), title
        )
        
        if success:
            comparisons_created += 1
            print(f"   ✅ Created: {comparison_path.name}")
        else:
            print(f"   ❌ Failed: {comparison_path.name}")
    
    return comparisons_created

def main():
    print("🎨 Generating visual comparisons and metrics charts")
    print("=" * 60)
    
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "experiments/zero_prior"
    base_path = Path(base_dir)
    
    # Generate comparison grid
    comparisons = generate_comparison_grid(base_dir)
    
    # Load metrics data
    metrics_path = base_path / "metrics" / "quantitative_results.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics_data = json.load(f)
    else:
        print("⚠️  No metrics data found - using mock data")
        from experiments.zero_prior.metrics import create_mock_analysis
        metrics_data = create_mock_analysis()
    
    # Generate metrics chart
    chart_path = base_path / "comparisons" / "metrics_chart.png"
    chart_success = create_metrics_chart(metrics_data, str(chart_path))
    
    print(f"\n✅ Comparison generation complete!")
    print(f"   📸 Image comparisons: {comparisons}")
    print(f"   📊 Metrics chart: {'✅' if chart_success else '❌'}")
    print(f"   📁 Output directory: {base_path / 'comparisons'}")

if __name__ == "__main__":
    main()