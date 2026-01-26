"""Video captioning model with transformer architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import (
    CLIPModel,
    CLIPProcessor,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)
from typing import Dict, List, Optional, Tuple, Union


class VideoEncoder(nn.Module):
    """Video encoder using CLIP visual encoder."""
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        freeze_weights: bool = True,
        hidden_dim: int = 512,
    ):
        super().__init__()
        
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        if freeze_weights:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        
        # Projection layer to match hidden dimension
        clip_dim = self.clip_model.config.vision_config.hidden_size
        self.projection = nn.Linear(clip_dim, hidden_dim)
        
    def forward(self, video_frames: torch.Tensor) -> torch.Tensor:
        """Encode video frames.
        
        Args:
            video_frames: Video tensor of shape (B, T, H, W, C).
            
        Returns:
            Encoded video features of shape (B, T, hidden_dim).
        """
        batch_size, num_frames = video_frames.shape[:2]
        
        # Reshape to process all frames at once
        frames_flat = video_frames.view(-1, *video_frames.shape[2:])
        
        # Process frames through CLIP
        with torch.no_grad() if not self.training else torch.enable_grad():
            # Convert to PIL Images for CLIP processor
            frames_pil = []
            for frame in frames_flat:
                # Convert tensor to PIL Image
                frame_np = (frame * 255).clamp(0, 255).byte().cpu().numpy()
                frame_pil = Image.fromarray(frame_np)
                frames_pil.append(frame_pil)
            
            # Process with CLIP
            inputs = self.processor(images=frames_pil, return_tensors="pt")
            inputs = {k: v.to(video_frames.device) for k, v in inputs.items()}
            
            vision_outputs = self.clip_model.vision_model(**inputs)
            frame_features = vision_outputs.pooler_output
        
        # Reshape back to (B, T, hidden_dim)
        frame_features = frame_features.view(batch_size, num_frames, -1)
        
        # Project to hidden dimension
        projected_features = self.projection(frame_features)
        
        return projected_features


class TextDecoder(nn.Module):
    """Text decoder using GPT-2."""
    
    def __init__(
        self,
        model_name: str = "gpt2",
        hidden_dim: int = 512,
        max_length: int = 128,
    ):
        super().__init__()
        
        self.gpt2_model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Projection layer to match GPT-2 hidden size
        gpt2_dim = self.gpt2_model.config.hidden_size
        self.projection = nn.Linear(hidden_dim, gpt2_dim)
        
        self.max_length = max_length
        
    def forward(
        self,
        video_features: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate text from video features.
        
        Args:
            video_features: Video features of shape (B, T, hidden_dim).
            input_ids: Input token IDs for teacher forcing.
            attention_mask: Attention mask for input tokens.
            
        Returns:
            Generated logits of shape (B, seq_len, vocab_size).
        """
        batch_size = video_features.shape[0]
        
        # Project video features to GPT-2 dimension
        projected_features = self.projection(video_features)
        
        # Create cross-attention inputs
        if input_ids is not None:
            # Teacher forcing mode
            outputs = self.gpt2_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=projected_features,
                use_cache=False,
            )
            return outputs.logits
        else:
            # Generation mode
            # Start with BOS token
            start_token = self.tokenizer.bos_token_id
            if start_token is None:
                start_token = self.tokenizer.eos_token_id
            
            input_ids = torch.full(
                (batch_size, 1),
                start_token,
                dtype=torch.long,
                device=video_features.device,
            )
            
            # Generate tokens
            for _ in range(self.max_length - 1):
                outputs = self.gpt2_model(
                    input_ids=input_ids,
                    encoder_hidden_states=projected_features,
                    use_cache=False,
                )
                
                # Get next token
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to input_ids
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Stop if all sequences hit EOS
                if (next_token == self.tokenizer.eos_token_id).all():
                    break
            
            return outputs.logits


class VideoCaptionTransformer(nn.Module):
    """Main video captioning model."""
    
    def __init__(
        self,
        video_encoder_config: Dict,
        text_decoder_config: Dict,
        fusion_type: str = "cross_attention",
        fusion_dim: int = 512,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-6,
    ):
        super().__init__()
        
        self.fusion_type = fusion_type
        self.fusion_dim = fusion_dim
        
        # Initialize components
        self.video_encoder = VideoEncoder(**video_encoder_config)
        self.text_decoder = TextDecoder(**text_decoder_config)
        
        # Fusion mechanism
        if fusion_type == "cross_attention":
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=fusion_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True,
            )
        elif fusion_type == "concatenation":
            self.fusion_projection = nn.Linear(
                fusion_dim * 2, fusion_dim
            )
        elif fusion_type == "bilinear":
            self.bilinear = nn.Bilinear(fusion_dim, fusion_dim, fusion_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(fusion_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        video_frames: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            video_frames: Video tensor of shape (B, T, H, W, C).
            input_ids: Input token IDs for teacher forcing.
            attention_mask: Attention mask for input tokens.
            
        Returns:
            Generated logits of shape (B, seq_len, vocab_size).
        """
        # Encode video
        video_features = self.video_encoder(video_frames)
        
        # Apply fusion if needed
        if self.fusion_type == "cross_attention":
            # Use cross-attention for fusion
            fused_features, _ = self.cross_attention(
                query=video_features,
                key=video_features,
                value=video_features,
            )
            fused_features = self.layer_norm(fused_features + video_features)
        else:
            fused_features = video_features
        
        fused_features = self.dropout(fused_features)
        
        # Generate text
        logits = self.text_decoder(
            video_features=fused_features,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        return logits
    
    def generate(
        self,
        video_frames: torch.Tensor,
        max_length: int = 128,
        num_beams: int = 5,
        temperature: float = 1.0,
        do_sample: bool = True,
        early_stopping: bool = True,
    ) -> List[str]:
        """Generate captions for video frames.
        
        Args:
            video_frames: Video tensor of shape (B, T, H, W, C).
            max_length: Maximum length of generated text.
            num_beams: Number of beams for beam search.
            temperature: Sampling temperature.
            do_sample: Whether to use sampling.
            early_stopping: Whether to use early stopping.
            
        Returns:
            List of generated captions.
        """
        self.eval()
        
        with torch.no_grad():
            # Encode video
            video_features = self.video_encoder(video_frames)
            
            # Apply fusion
            if self.fusion_type == "cross_attention":
                fused_features, _ = self.cross_attention(
                    query=video_features,
                    key=video_features,
                    value=video_features,
                )
                fused_features = self.layer_norm(fused_features + video_features)
            else:
                fused_features = video_features
            
            # Generate text using the decoder's generation method
            generated_ids = self.text_decoder.gpt2_model.generate(
                encoder_hidden_states=fused_features,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                do_sample=do_sample,
                early_stopping=early_stopping,
                pad_token_id=self.text_decoder.tokenizer.pad_token_id,
                eos_token_id=self.text_decoder.tokenizer.eos_token_id,
            )
            
            # Decode to text
            captions = []
            for ids in generated_ids:
                caption = self.text_decoder.tokenizer.decode(
                    ids, skip_special_tokens=True
                )
                captions.append(caption)
            
            return captions
