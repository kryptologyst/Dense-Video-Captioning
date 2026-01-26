#!/usr/bin/env python3
"""Script to create a toy dataset for testing."""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.video_caption_dataset import create_toy_dataset


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Create toy dataset")
    parser.add_argument("--output_dir", type=str, default="data/toy",
                       help="Output directory for toy dataset")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples to generate")
    parser.add_argument("--video_duration", type=float, default=5.0,
                       help="Duration of each video in seconds")
    parser.add_argument("--fps", type=float, default=30.0,
                       help="Frames per second for generated videos")
    
    args = parser.parse_args()
    
    print(f"Creating toy dataset with {args.num_samples} samples...")
    print(f"Output directory: {args.output_dir}")
    print(f"Video duration: {args.video_duration}s")
    print(f"FPS: {args.fps}")
    
    create_toy_dataset(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        video_duration=args.video_duration,
        fps=args.fps,
    )
    
    print("Toy dataset created successfully!")


if __name__ == "__main__":
    main()
