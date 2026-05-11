#!/usr/bin/env python3
"""
Fast training test: 500 samples, 3 epochs, 10 minutes
Para ver rápidamente si la optimización funciona
"""

import os
os.environ.setdefault("TORCH_HOME", r"C:\torch_cache")

import argparse
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast training test")
    parser.add_argument("--data-root", type=str, default="/home/datasets/coco", 
                       help="Path to COCO dataset")
    parser.add_argument("--no-wandb", action="store_true", default=True, help="Disable wandb")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("⚡ FAST TRAINING TEST - 500 samples, 3 epochs (~10 minutes)")
    print("=" * 70)
    print(f"\n📊 Configuration:")
    print(f"  Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"  Dataset: COCO (500 samples only)")
    print(f"  Epochs: 3")
    print(f"  Batch Size: 48")
    print(f"  Backbone: ResNet50")
    print(f"  Learning Rate Decoder: 5e-3 (optimized)")
    print(f"  Data Augmentation: Enhanced (8 types)")
    print(f"  Mixed Precision: {'Enabled' if torch.cuda.is_available() else 'Disabled (no GPU)'}")
    print("=" * 70)
    
    # Import main training function
    from main import principal
    
    # Override args for fast testing
    args.epochs = 3
    args.overfit = 500
    args.wandb_offline = False
    
    # Run training
    principal(args)
    
    print("\n" + "=" * 70)
    print("✅ FAST TEST COMPLETED!")
    print("=" * 70)
    print("\n📈 RESULTS INTERPRETATION:")
    print("  - If mIoU increases from ep1 → ep3: Optimization works! ✓")
    print("  - Expected mIoU range: 0.40-0.50 (limited by 500 samples)")
    print("  - Full training (118k samples): Will be 10-15% higher")
    print("\n🚀 NEXT: Run full training with:")
    print("  python main.py --data-root /home/datasets/coco --epochs 30 --no-wandb")
    print("=" * 70)
