"""Evaluation script for dense video captioning."""

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
from data.video_caption_dataset import VideoCaptionDataset, create_dataloader
from utils.device import set_seed, get_device
from utils.metrics import calculate_metrics, print_metrics


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate dense video captioning model")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="test",
                       choices=["train", "val", "test"],
                       help="Dataset split to evaluate")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cuda, mps, cpu)")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Path to save evaluation results")
    
    args = parser.parse_args()
    
    # Load config
    config = OmegaConf.load(args.config)
    
    # Set device
    device = get_device(args.device)
    config.device = str(device)
    
    # Set seed for reproducibility
    set_seed(config.seed, config.deterministic)
    
    print(f"Using device: {device}")
    print(f"Evaluating on {args.split} split")
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset
    print("Loading dataset...")
    
    if args.split == "train":
        video_dir = config.data.train_video_dir
        annotation_file = config.data.train_annotation_file
    elif args.split == "val":
        video_dir = config.data.val_video_dir
        annotation_file = config.data.val_annotation_file
    else:  # test
        video_dir = config.data.test_video_dir
        annotation_file = config.data.test_annotation_file
    
    dataset = VideoCaptionDataset(
        video_dir=video_dir,
        annotation_file=annotation_file,
        tokenizer=tokenizer,
        video_fps=config.data.video_fps,
        max_frames=config.data.max_frames,
        frame_size=tuple(config.data.frame_size),
        max_caption_length=config.data.max_caption_length,
        use_augmentation=False,  # No augmentation for evaluation
    )
    
    # Create data loader
    data_loader = create_dataloader(
        dataset=dataset,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        shuffle=False,
        pin_memory=config.data.pin_memory,
    )
    
    print(f"Dataset samples: {len(dataset)}")
    
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
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
    
    # Evaluate model
    print("Evaluating model...")
    
    all_predictions = []
    all_targets = []
    all_video_files = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            print(f"Processing batch {batch_idx + 1}/{len(data_loader)}")
            
            # Move batch to device
            video_frames = batch['video_frames'].to(device)
            
            # Generate predictions
            predictions = model.generate(
                video_frames=video_frames,
                max_length=config.data.max_caption_length,
                num_beams=5,
                temperature=1.0,
                do_sample=False,
                early_stopping=True,
            )
            
            all_predictions.extend(predictions)
            all_targets.extend(batch['captions'])
            all_video_files.extend(batch['video_files'])
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics = calculate_metrics(all_predictions, all_targets)
    
    # Print metrics
    print_metrics(metrics)
    
    # Save results if specified
    if args.output_file:
        results = {
            'metrics': metrics,
            'predictions': all_predictions,
            'targets': all_targets,
            'video_files': all_video_files,
            'checkpoint': args.checkpoint,
            'split': args.split,
        }
        
        torch.save(results, args.output_file)
        print(f"Results saved to: {args.output_file}")
    
    # Print some example predictions
    print("\n" + "="*50)
    print("EXAMPLE PREDICTIONS")
    print("="*50)
    
    for i in range(min(5, len(all_predictions))):
        print(f"\nVideo: {all_video_files[i]}")
        print(f"Target: {all_targets[i]}")
        print(f"Prediction: {all_predictions[i]}")
        print("-" * 30)
    
    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()
