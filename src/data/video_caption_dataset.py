"""Dataset classes for video captioning."""

import json
import os
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer

from ..utils.video import extract_frames_from_video, frames_to_tensor


class VideoCaptionDataset(Dataset):
    """Dataset for video captioning tasks."""
    
    def __init__(
        self,
        video_dir: str,
        annotation_file: str,
        tokenizer: GPT2Tokenizer,
        video_fps: float = 1.0,
        max_frames: int = 32,
        frame_size: Tuple[int, int] = (224, 224),
        max_caption_length: int = 128,
        use_augmentation: bool = True,
        augmentation_prob: float = 0.5,
    ):
        self.video_dir = video_dir
        self.tokenizer = tokenizer
        self.video_fps = video_fps
        self.max_frames = max_frames
        self.frame_size = frame_size
        self.max_caption_length = max_caption_length
        self.use_augmentation = use_augmentation
        self.augmentation_prob = augmentation_prob
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Filter valid samples
        self.samples = []
        for ann in self.annotations:
            video_path = os.path.join(video_dir, ann['video_file'])
            if os.path.exists(video_path):
                self.samples.append(ann)
        
        print(f"Loaded {len(self.samples)} valid samples from {annotation_file}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load video frames
        video_path = os.path.join(self.video_dir, sample['video_file'])
        frames = extract_frames_from_video(
            video_path,
            fps=self.video_fps,
            max_frames=self.max_frames,
            frame_size=self.frame_size,
        )
        
        # Convert to tensor
        video_tensor = frames_to_tensor(frames)
        
        # Apply augmentation if enabled
        if self.use_augmentation and np.random.random() < self.augmentation_prob:
            video_tensor = self._apply_augmentation(video_tensor)
        
        # Process caption
        caption = sample['caption']
        caption_tokens = self.tokenizer.encode(
            caption,
            max_length=self.max_caption_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        ).squeeze(0)
        
        # Create attention mask
        attention_mask = (caption_tokens != self.tokenizer.pad_token_id).long()
        
        return {
            'video_frames': video_tensor,
            'caption_tokens': caption_tokens,
            'attention_mask': attention_mask,
            'video_file': sample['video_file'],
            'caption': caption,
        }
    
    def _apply_augmentation(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation to video tensor.
        
        Args:
            video_tensor: Video tensor of shape (T, H, W, C).
            
        Returns:
            Augmented video tensor.
        """
        # Random horizontal flip
        if np.random.random() < 0.5:
            video_tensor = torch.flip(video_tensor, dims=[2])
        
        # Color jitter
        if np.random.random() < 0.3:
            # Brightness
            brightness_factor = 1.0 + np.random.uniform(-0.2, 0.2)
            video_tensor = torch.clamp(video_tensor * brightness_factor, 0, 1)
            
            # Contrast
            contrast_factor = 1.0 + np.random.uniform(-0.2, 0.2)
            mean = video_tensor.mean()
            video_tensor = torch.clamp(
                (video_tensor - mean) * contrast_factor + mean, 0, 1
            )
        
        # Random crop (if frames are larger than target size)
        if video_tensor.shape[1] > self.frame_size[0] or video_tensor.shape[2] > self.frame_size[1]:
            h, w = video_tensor.shape[1:3]
            target_h, target_w = self.frame_size
            
            if h > target_h:
                start_h = np.random.randint(0, h - target_h + 1)
                video_tensor = video_tensor[:, start_h:start_h + target_h, :, :]
            
            if w > target_w:
                start_w = np.random.randint(0, w - target_w + 1)
                video_tensor = video_tensor[:, :, start_w:start_w + target_w, :]
        
        return video_tensor


def create_dataloader(
    dataset: VideoCaptionDataset,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader for the dataset.
    
    Args:
        dataset: Video captioning dataset.
        batch_size: Batch size.
        num_workers: Number of worker processes.
        shuffle: Whether to shuffle the data.
        pin_memory: Whether to pin memory.
        
    Returns:
        DataLoader instance.
    """
    def collate_fn(batch):
        """Custom collate function for video captioning."""
        video_frames = torch.stack([item['video_frames'] for item in batch])
        caption_tokens = torch.stack([item['caption_tokens'] for item in batch])
        attention_masks = torch.stack([item['attention_mask'] for item in batch])
        
        return {
            'video_frames': video_frames,
            'caption_tokens': caption_tokens,
            'attention_masks': attention_masks,
            'video_files': [item['video_file'] for item in batch],
            'captions': [item['caption'] for item in batch],
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )


def create_toy_dataset(
    output_dir: str = "data/toy",
    num_samples: int = 100,
    video_duration: float = 5.0,
    fps: float = 30.0,
) -> None:
    """Create a toy dataset for testing.
    
    Args:
        output_dir: Output directory for the toy dataset.
        num_samples: Number of samples to generate.
        video_duration: Duration of each video in seconds.
        fps: Frames per second for generated videos.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "videos"), exist_ok=True)
    
    # Sample captions
    captions = [
        "A person walking in a park",
        "A dog running on grass",
        "A car driving on a street",
        "A bird flying in the sky",
        "A cat sitting on a chair",
        "A person cooking in a kitchen",
        "A child playing with toys",
        "A person reading a book",
        "A dog playing with a ball",
        "A person exercising at home",
    ]
    
    annotations = []
    
    for i in range(num_samples):
        # Generate random video
        video_path = os.path.join(output_dir, "videos", f"video_{i:04d}.mp4")
        
        # Create a simple colored video
        frame_count = int(video_duration * fps)
        height, width = 224, 224
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        # Generate frames with random colors and simple patterns
        for frame_idx in range(frame_count):
            # Create a frame with random background color
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            
            # Add some simple patterns
            if frame_idx % 30 < 15:  # Moving rectangle
                x = int((frame_idx % 30) * width / 15)
                cv2.rectangle(frame, (x, 50), (x + 50, 100), (255, 255, 255), -1)
            
            out.write(frame)
        
        out.release()
        
        # Add annotation
        caption = captions[i % len(captions)]
        annotations.append({
            'video_file': f"video_{i:04d}.mp4",
            'caption': caption,
        })
    
    # Save annotations
    with open(os.path.join(output_dir, "annotations.json"), 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"Created toy dataset with {num_samples} samples in {output_dir}")
