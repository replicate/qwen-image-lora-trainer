#!/usr/bin/env python3
"""
Comprehensive zero-prior placeholder token experiment runner.
Executes controlled training and inference comparisons.
"""

import os
import sys
import json
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import time

class ExperimentRunner:
    def __init__(self, base_dir: str = "experiments/zero_prior"):
        self.base_dir = Path(base_dir)
        self.artifacts_dir = self.base_dir / "artifacts"
        self.outputs_dir = self.base_dir / "outputs"
        self.datasets_dir = self.base_dir / "datasets"
        self.configs_dir = self.base_dir / "configs"
        
        # Ensure directories exist
        for dir_path in [self.artifacts_dir, self.outputs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def load_config(self) -> Dict:
        """Load common configuration"""
        config_path = self.configs_dir / "common.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f)
    
    def run_training(self, mode: str, dataset_zip: str) -> Tuple[bool, str, str]:
        """Run Cog training for baseline or zero-prior mode"""
        config = self.load_config()
        
        # Determine training parameters
        if mode == "baseline":
            train_placeholder_token = False
            output_name = "baseline_weights.zip"
        elif mode == "zeropri":
            train_placeholder_token = True  
            output_name = "zeropri_weights.zip"
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        output_path = self.artifacts_dir / output_name
        
        print(f"🚀 Starting {mode} training...")
        print(f"   Dataset: {dataset_zip}")
        print(f"   Output: {output_path}")
        
        # Build Cog command
        cmd = [
            "cog", "train",
            "-i", f"dataset=@{dataset_zip}",
            "-i", f"train_placeholder_token={str(train_placeholder_token).lower()}",
            "-i", f"rank={config['training']['rank']}",
            "-i", f"steps={config['training']['steps']}",
            "-i", f"learning_rate={config['training']['learning_rate']}",
            "-i", f"batch_size={config['training']['batch_size']}",
        ]
        
        # Add zero-prior specific parameters
        if train_placeholder_token:
            cmd.extend([
                "-i", f"placeholder_token={config['model']['placeholder_token']}",
                "-i", f"placeholder_init_std={config['model']['placeholder_init_std']}"
            ])
        
        # Execute training
        start_time = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
            success = result.returncode == 0
            duration = time.time() - start_time
            
            # Move output to artifacts directory
            if success and Path("weights.zip").exists():
                shutil.move("weights.zip", output_path)
                print(f"   ✅ Training completed in {duration:.1f}s")
                print(f"   📦 Weights saved: {output_path}")
            else:
                print(f"   ❌ Training failed after {duration:.1f}s")
                print(f"   Error: {result.stderr}")
            
            return success, str(output_path), result.stderr
            
        except subprocess.TimeoutExpired:
            print(f"   ⏰ Training timed out after 30 minutes")
            return False, "", "Training timeout"
    
    def run_inference_sweep(self, mode: str, weights_path: str) -> List[str]:
        """Run inference sweep with evaluation prompts"""
        config = self.load_config()
        eval_prompts = config['experiment']['eval_prompts'][mode]
        
        output_images = []
        inference_params = config['experiment']['inference_params']
        
        print(f"🖼️  Running {mode} inference sweep...")
        
        for i, prompt in enumerate(eval_prompts):
            output_name = f"{mode}_image_{i+1:02d}.png"
            output_path = self.outputs_dir / output_name
            
            print(f"   Prompt {i+1}: {prompt}")
            
            # Build Cog prediction command
            cmd = [
                "cog", "predict",
                "-i", f"prompt={prompt}",
                "-i", f"replicate_weights=@{weights_path}",
                "-i", f"width={inference_params['width']}",
                "-i", f"height={inference_params['height']}",
                "-i", f"num_inference_steps={inference_params['num_inference_steps']}",
                "-i", f"guidance_scale={inference_params['guidance_scale']}",
                "-i", f"seed={inference_params['seed']}",
                "-o", str(output_path)
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if result.returncode == 0 and output_path.exists():
                    output_images.append(str(output_path))
                    print(f"     ✅ Generated: {output_name}")
                else:
                    print(f"     ❌ Failed: {result.stderr}")
            
            except subprocess.TimeoutExpired:
                print(f"     ⏰ Inference timeout")
        
        print(f"   📸 Generated {len(output_images)} images")
        return output_images
    
    def create_experiment_log(self, results: Dict) -> str:
        """Create detailed experiment log"""
        log_path = self.base_dir / "experiment_log.json"
        
        log_data = {
            "experiment_id": f"zero_prior_{int(time.time())}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": self.load_config(),
            "results": results,
            "system_info": {
                "platform": os.name,
                "python_version": sys.version,
                "working_directory": os.getcwd()
            }
        }
        
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        return str(log_path)
    
    def run_full_experiment(self, dataset_zip: str) -> Dict:
        """Run complete baseline vs zero-prior experiment"""
        print("🧪 Starting comprehensive zero-prior token experiment")
        print("=" * 60)
        
        results = {
            "baseline": {"training": {}, "inference": {}},
            "zeropri": {"training": {}, "inference": {}}
        }
        
        # Prepare datasets
        print("\n📋 Preparing datasets...")
        baseline_dataset = self.datasets_dir / "baseline_dataset.zip"
        zeropri_dataset = self.datasets_dir / "zeropri_dataset.zip"
        
        if not baseline_dataset.exists() or not zeropri_dataset.exists():
            print("   Generating captions...")
            subprocess.run([
                "python", str(self.base_dir / "make_captions.py"), 
                dataset_zip
            ])
        
        # Run baseline training
        print(f"\n🏁 Phase 1: Baseline Training")
        success, weights_path, error = self.run_training("baseline", str(baseline_dataset))
        results["baseline"]["training"] = {
            "success": success,
            "weights_path": weights_path,
            "error": error
        }
        
        if success:
            # Run baseline inference
            print(f"\n🏁 Phase 2: Baseline Inference")
            images = self.run_inference_sweep("baseline", weights_path)
            results["baseline"]["inference"] = {
                "images": images,
                "count": len(images)
            }
        
        # Run zero-prior training
        print(f"\n🔮 Phase 3: Zero-Prior Training")
        success, weights_path, error = self.run_training("zeropri", str(zeropri_dataset))
        results["zeropri"]["training"] = {
            "success": success,
            "weights_path": weights_path,
            "error": error
        }
        
        if success:
            # Run zero-prior inference
            print(f"\n🔮 Phase 4: Zero-Prior Inference")
            images = self.run_inference_sweep("zeropri", weights_path)
            results["zeropri"]["inference"] = {
                "images": images,
                "count": len(images)
            }
        
        # Create experiment log
        log_path = self.create_experiment_log(results)
        print(f"\n📋 Experiment log: {log_path}")
        
        return results

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_experiment.py <dataset.zip>")
        print("  Runs complete baseline vs zero-prior comparison")
        sys.exit(1)
    
    dataset_zip = sys.argv[1]
    if not Path(dataset_zip).exists():
        print(f"Error: Dataset {dataset_zip} not found")
        sys.exit(1)
    
    runner = ExperimentRunner()
    results = runner.run_full_experiment(dataset_zip)
    
    # Print summary
    print("\n📊 Experiment Summary:")
    print(f"   Baseline training: {'✅' if results['baseline']['training']['success'] else '❌'}")
    print(f"   Baseline inference: {results['baseline']['inference'].get('count', 0)} images")
    print(f"   Zero-prior training: {'✅' if results['zeropri']['training']['success'] else '❌'}")
    print(f"   Zero-prior inference: {results['zeropri']['inference'].get('count', 0)} images")
    
    print(f"\n🎯 Next steps:")
    print(f"   1. Run: python experiments/zero_prior/metrics.py")
    print(f"   2. Review: experiments/zero_prior/TEST_EXPERIMENT.md")

if __name__ == "__main__":
    main()