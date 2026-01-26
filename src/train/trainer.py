"""Training utilities for video captioning."""

import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from ..utils.device import get_device, get_model_size, get_device_memory_info
from ..utils.metrics import calculate_metrics


class VideoCaptionTrainer:
    """Trainer class for video captioning models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tokenizer,
        config: Dict,
        device: Optional[str] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.config = config
        self.device = get_device(device)
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5),
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('max_epochs', 100),
            eta_min=config.get('min_lr', 1e-6),
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        
        # Mixed precision
        self.use_amp = config.get('mixed_precision', True)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Logging
        self.log_dir = config.get('log_dir', 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        
        # Checkpointing
        self.checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_metrics = {}
        
        # Print model info
        model_info = get_model_size(self.model)
        print(f"Model parameters: {model_info['total_parameters_millions']:.2f}M")
        print(f"Trainable parameters: {model_info['trainable_parameters_millions']:.2f}M")
        print(f"Device: {self.device}")
        print(f"Memory info: {get_device_memory_info(self.device)}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary containing training metrics.
        """
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            video_frames = batch['video_frames'].to(self.device)
            caption_tokens = batch['caption_tokens'].to(self.device)
            attention_masks = batch['attention_masks'].to(self.device)
            
            # Forward pass
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logits = self.model(
                        video_frames=video_frames,
                        input_ids=caption_tokens[:, :-1],  # Remove last token
                        attention_mask=attention_masks[:, :-1],
                    )
                    
                    # Calculate loss
                    targets = caption_tokens[:, 1:]  # Remove first token
                    loss = self.criterion(
                        logits.reshape(-1, logits.size(-1)),
                        targets.reshape(-1),
                    )
            else:
                logits = self.model(
                    video_frames=video_frames,
                    input_ids=caption_tokens[:, :-1],
                    attention_mask=attention_masks[:, :-1],
                )
                
                targets = caption_tokens[:, 1:]
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                )
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_samples += video_frames.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / (batch_idx + 1):.4f}",
            })
            
            # Log to tensorboard
            if batch_idx % 100 == 0:
                self.writer.add_scalar(
                    'Train/Loss',
                    loss.item(),
                    self.current_epoch * len(self.train_loader) + batch_idx,
                )
        
        avg_loss = total_loss / len(self.train_loader)
        
        return {
            'train_loss': avg_loss,
            'train_samples': total_samples,
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate the model.
        
        Returns:
            Dictionary containing validation metrics.
        """
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            
            for batch in pbar:
                # Move batch to device
                video_frames = batch['video_frames'].to(self.device)
                caption_tokens = batch['caption_tokens'].to(self.device)
                attention_masks = batch['attention_masks'].to(self.device)
                
                # Forward pass
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        logits = self.model(
                            video_frames=video_frames,
                            input_ids=caption_tokens[:, :-1],
                            attention_mask=attention_masks[:, :-1],
                        )
                        
                        targets = caption_tokens[:, 1:]
                        loss = self.criterion(
                            logits.reshape(-1, logits.size(-1)),
                            targets.reshape(-1),
                        )
                else:
                    logits = self.model(
                        video_frames=video_frames,
                        input_ids=caption_tokens[:, :-1],
                        attention_mask=attention_masks[:, :-1],
                    )
                    
                    targets = caption_tokens[:, 1:]
                    loss = self.criterion(
                        logits.reshape(-1, logits.size(-1)),
                        targets.reshape(-1),
                    )
                
                total_loss += loss.item()
                
                # Generate predictions for evaluation
                predictions = self.model.generate(
                    video_frames=video_frames,
                    max_length=self.config.get('max_caption_length', 128),
                )
                
                all_predictions.extend(predictions)
                all_targets.extend(batch['captions'])
                
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(self.val_loader)
        
        # Calculate metrics
        metrics = calculate_metrics(all_predictions, all_targets)
        metrics['val_loss'] = avg_loss
        
        return metrics
    
    def train(self, max_epochs: int = 100, save_every: int = 10) -> None:
        """Train the model.
        
        Args:
            max_epochs: Maximum number of epochs to train.
            save_every: Save checkpoint every N epochs.
        """
        print(f"Starting training for {max_epochs} epochs...")
        
        for epoch in range(max_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Log metrics
            self.writer.add_scalar('Train/Loss', train_metrics['train_loss'], epoch)
            self.writer.add_scalar('Val/Loss', val_metrics['val_loss'], epoch)
            
            for metric_name, metric_value in val_metrics.items():
                if metric_name != 'val_loss':
                    self.writer.add_scalar(f'Val/{metric_name}', metric_value, epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{max_epochs}")
            print(f"Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"Val Metrics: {val_metrics}")
            
            # Save best model
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.best_val_metrics = val_metrics
                self.save_checkpoint('best_model.pth')
                print(f"New best model saved! Val Loss: {self.best_val_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')
        
        print(f"\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best validation metrics: {self.best_val_metrics}")
    
    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint.
        
        Args:
            filename: Checkpoint filename.
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_metrics': self.best_val_metrics,
            'config': self.config,
        }
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint.
        
        Args:
            filename: Checkpoint filename.
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_metrics = checkpoint['best_val_metrics']
        
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"Resuming from epoch {self.current_epoch + 1}")
    
    def close(self) -> None:
        """Close the trainer and cleanup resources."""
        self.writer.close()
