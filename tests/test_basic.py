"""Tests for dense video captioning."""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.video_caption_transformer import VideoCaptionTransformer
from utils.device import get_device, set_seed
from utils.video import extract_frames_from_video, frames_to_tensor
from utils.metrics import calculate_metrics


class TestVideoCaptionTransformer:
    """Test cases for VideoCaptionTransformer."""
    
    def test_model_initialization(self):
        """Test model initialization."""
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
        
        assert model is not None
        assert hasattr(model, 'video_encoder')
        assert hasattr(model, 'text_decoder')
        assert hasattr(model, 'cross_attention')
    
    def test_model_forward(self):
        """Test model forward pass."""
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
        
        # Create dummy video tensor
        batch_size = 2
        num_frames = 8
        height, width, channels = 224, 224, 3
        
        video_tensor = torch.randn(batch_size, num_frames, height, width, channels)
        
        # Test forward pass
        with torch.no_grad():
            output = model(video_tensor)
        
        assert output is not None
        assert output.shape[0] == batch_size  # Batch dimension
    
    def test_model_generation(self):
        """Test model generation."""
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
        
        # Create dummy video tensor
        batch_size = 1
        num_frames = 8
        height, width, channels = 224, 224, 3
        
        video_tensor = torch.randn(batch_size, num_frames, height, width, channels)
        
        # Test generation
        with torch.no_grad():
            captions = model.generate(video_tensor, max_length=50)
        
        assert isinstance(captions, list)
        assert len(captions) == batch_size
        assert isinstance(captions[0], str)


class TestDeviceUtils:
    """Test cases for device utilities."""
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device("auto")
        assert device is not None
        assert isinstance(device, torch.device)
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42, deterministic=True)
        
        # Check that random numbers are deterministic
        torch.manual_seed(42)
        rand1 = torch.randn(10)
        
        torch.manual_seed(42)
        rand2 = torch.randn(10)
        
        assert torch.allclose(rand1, rand2)


class TestVideoUtils:
    """Test cases for video utilities."""
    
    def test_frames_to_tensor(self):
        """Test frame to tensor conversion."""
        # Create dummy frames
        frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(8)]
        
        tensor = frames_to_tensor(frames)
        
        assert tensor.shape == (8, 224, 224, 3)
        assert tensor.dtype == torch.float32
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0
    
    def test_tensor_to_frames(self):
        """Test tensor to frames conversion."""
        # Create dummy tensor
        tensor = torch.rand(8, 224, 224, 3)
        
        frames = tensor_to_frames(tensor)
        
        assert len(frames) == 8
        assert all(frame.shape == (224, 224, 3) for frame in frames)
        assert all(frame.dtype == np.uint8 for frame in frames)


class TestMetrics:
    """Test cases for metrics."""
    
    def test_calculate_metrics(self):
        """Test metrics calculation."""
        predictions = [
            "A person walking in a park",
            "A dog running on grass",
            "A car driving on a street",
        ]
        references = [
            "A person walking in a park",
            "A dog running on grass", 
            "A car driving on a street",
        ]
        
        metrics = calculate_metrics(predictions, references)
        
        assert isinstance(metrics, dict)
        assert "exact_match" in metrics
        assert "bleu_1" in metrics
        assert "meteor" in metrics
        
        # Perfect predictions should have high scores
        assert metrics["exact_match"] == 1.0
        assert metrics["bleu_1"] > 0.8
    
    def test_normalize_text(self):
        """Test text normalization."""
        from utils.metrics import normalize_text
        
        text = "  A Person Walking in a Park!!!  "
        normalized = normalize_text(text)
        
        assert normalized == "a person walking in a park"


if __name__ == "__main__":
    pytest.main([__file__])
