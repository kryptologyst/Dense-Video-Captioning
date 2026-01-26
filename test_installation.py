#!/usr/bin/env python3
"""Test script to verify installation and basic functionality."""

import sys
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"✗ Transformers import failed: {e}")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import PIL
        print(f"✓ Pillow {PIL.__version__}")
    except ImportError as e:
        print(f"✗ Pillow import failed: {e}")
        return False
    
    try:
        import omegaconf
        print(f"✓ OmegaConf {omegaconf.__version__}")
    except ImportError as e:
        print(f"✗ OmegaConf import failed: {e}")
        return False
    
    return True


def test_device():
    """Test device detection."""
    print("\nTesting device detection...")
    
    try:
        from src.utils.device import get_device
        device = get_device("auto")
        print(f"✓ Device detected: {device}")
        return True
    except Exception as e:
        print(f"✗ Device detection failed: {e}")
        return False


def test_model():
    """Test model initialization."""
    print("\nTesting model initialization...")
    
    try:
        from src.models.video_caption_transformer import VideoCaptionTransformer
        
        model = VideoCaptionTransformer(
            video_encoder_config={
                "model_name": "openai/clip-vit-base-patch32",
                "freeze_weights": True,
                "hidden_dim": 512,
            },
            text_decoder_config={
                "model_name": "gpt2",
                "hidden_dim": 512,
                "max_length": 128,
            },
            fusion_type="cross_attention",
            fusion_dim=512,
            dropout=0.1,
        )
        
        print(f"✓ Model initialized successfully")
        print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  - Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        return True
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        return False


def test_data_creation():
    """Test toy dataset creation."""
    print("\nTesting toy dataset creation...")
    
    try:
        from src.data.video_caption_dataset import create_toy_dataset
        
        create_toy_dataset(
            output_dir="test_data",
            num_samples=5,
            video_duration=2.0,
            fps=30.0,
        )
        
        print("✓ Toy dataset created successfully")
        return True
    except Exception as e:
        print(f"✗ Toy dataset creation failed: {e}")
        return False


def test_metrics():
    """Test metrics calculation."""
    print("\nTesting metrics calculation...")
    
    try:
        from src.utils.metrics import calculate_metrics
        
        predictions = ["A person walking in a park", "A dog running on grass"]
        references = ["A person walking in a park", "A dog running on grass"]
        
        metrics = calculate_metrics(predictions, references)
        
        print("✓ Metrics calculated successfully")
        print(f"  - Exact match: {metrics['exact_match']:.3f}")
        print(f"  - BLEU-1: {metrics['bleu_1']:.3f}")
        print(f"  - METEOR: {metrics['meteor']:.3f}")
        return True
    except Exception as e:
        print(f"✗ Metrics calculation failed: {e}")
        return False


def main():
    """Main test function."""
    print("=" * 50)
    print("DENSE VIDEO CAPTIONING - INSTALLATION TEST")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_device,
        test_model,
        test_data_creation,
        test_metrics,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 50)
    
    if passed == total:
        print("🎉 All tests passed! Installation is successful.")
        print("\nNext steps:")
        print("1. Create a toy dataset: python scripts/create_toy_dataset.py")
        print("2. Train a model: python src/train/main.py --create_toy_data")
        print("3. Run the demo: python scripts/run_demo.py")
        return 0
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
