"""Main training script for dense video captioning."""

import argparse
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from transformers import GPT2Tokenizer

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.video_caption_transformer import VideoCaptionTransformer
from data.video_caption_dataset import VideoCaptionDataset, create_dataloader, create_toy_dataset
from train.trainer import VideoCaptionTrainer
from utils.device import set_seed, get_device


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train dense video captioning model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", 
                       help="Path to config file")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--create_toy_data", action="store_true",
                       help="Create toy dataset for testing")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cuda, mps, cpu)")
    
    args = parser.parse_args()
    
    # Load config
    config = OmegaConf.load(args.config)
    
    # Set device
    device = get_device(args.device)
    config.device = str(device)
    
    # Set seed for reproducibility
    set_seed(config.seed, config.deterministic)
    
    print(f"Using device: {device}")
    print(f"Config: {OmegaConf.to_yaml(config)}")
    
    # Create toy dataset if requested
    if args.create_toy_data:
        print("Creating toy dataset...")
        create_toy_dataset(
            output_dir="data/toy",
            num_samples=100,
            video_duration=5.0,
            fps=30.0,
        )
        
        # Update config to use toy dataset
        config.data.train_video_dir = "data/toy/videos"
        config.data.train_annotation_file = "data/toy/annotations.json"
        config.data.val_video_dir = "data/toy/videos"
        config.data.val_annotation_file = "data/toy/annotations.json"
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets
    print("Loading datasets...")
    
    train_dataset = VideoCaptionDataset(
        video_dir=config.data.train_video_dir,
        annotation_file=config.data.train_annotation_file,
        tokenizer=tokenizer,
        video_fps=config.data.video_fps,
        max_frames=config.data.max_frames,
        frame_size=tuple(config.data.frame_size),
        max_caption_length=config.data.max_caption_length,
        use_augmentation=config.data.use_augmentation,
        augmentation_prob=config.data.augmentation_prob,
    )
    
    val_dataset = VideoCaptionDataset(
        video_dir=config.data.val_video_dir,
        annotation_file=config.data.val_annotation_file,
        tokenizer=tokenizer,
        video_fps=config.data.video_fps,
        max_frames=config.data.max_frames,
        frame_size=tuple(config.data.frame_size),
        max_caption_length=config.data.max_caption_length,
        use_augmentation=False,  # No augmentation for validation
    )
    
    # Create data loaders
    train_loader = create_dataloader(
        dataset=train_dataset,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        shuffle=True,
        pin_memory=config.data.pin_memory,
    )
    
    val_loader = create_dataloader(
        dataset=val_dataset,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        shuffle=False,
        pin_memory=config.data.pin_memory,
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Initialize model
    print("Initializing model...")
    model = VideoCaptionTransformer(
        video_encoder_config={
            "model_name": config.model.video_encoder.model_name,
            "freeze_weights": config.model.video_encoder.freeze_weights,
            "hidden_dim": config.model.hidden_dim,
        },
        text_decoder_config={
            "model_name": config.model.text_decoder.model_name,
            "hidden_dim": config.model.hidden_dim,
            "max_length": config.model.text_decoder.max_length,
        },
        fusion_type=config.model.fusion_type,
        fusion_dim=config.model.fusion_dim,
        dropout=config.model.dropout,
        layer_norm_eps=config.model.layer_norm_eps,
    )
    
    # Initialize trainer
    trainer = VideoCaptionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        config=config,
        device=device,
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train the model
    trainer.train(
        max_epochs=config.training.max_epochs,
        save_every=config.training.save_every,
    )
    
    # Close trainer
    trainer.close()
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
