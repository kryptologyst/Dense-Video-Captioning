"""Streamlit demo for dense video captioning."""

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from transformers import GPT2Tokenizer

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.video_caption_transformer import VideoCaptionTransformer
from utils.device import get_device
from utils.video import extract_frames_from_video, get_video_info


@st.cache_resource
def load_model(checkpoint_path: str, config_path: str):
    """Load the trained model."""
    # Load config
    config = OmegaConf.load(config_path)
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model
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
    device = get_device("auto")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, tokenizer, device, config


def process_video(video_file, config):
    """Process uploaded video file."""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_file.read())
        tmp_path = tmp_file.name
    
    try:
        # Extract frames
        frames = extract_frames_from_video(
            tmp_path,
            fps=config.data.video_fps,
            max_frames=config.data.max_frames,
            frame_size=tuple(config.data.frame_size),
        )
        
        # Convert to tensor
        video_tensor = torch.stack([
            torch.from_numpy(frame).float() / 255.0 
            for frame in frames
        ])
        
        # Add batch dimension
        video_tensor = video_tensor.unsqueeze(0)
        
        return video_tensor, frames
        
    finally:
        # Clean up temporary file
        os.unlink(tmp_path)


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Dense Video Captioning",
        page_icon="🎬",
        layout="wide",
    )
    
    st.title("🎬 Dense Video Captioning")
    st.markdown("Generate captions for videos using advanced computer vision models")
    
    # Sidebar for model selection
    st.sidebar.header("Model Configuration")
    
    # Check if model files exist
    checkpoint_path = "checkpoints/best_model.pth"
    config_path = "configs/config.yaml"
    
    if not os.path.exists(checkpoint_path):
        st.error(f"Model checkpoint not found: {checkpoint_path}")
        st.info("Please train a model first or provide a valid checkpoint path.")
        return
    
    if not os.path.exists(config_path):
        st.error(f"Config file not found: {config_path}")
        return
    
    # Load model
    with st.spinner("Loading model..."):
        try:
            model, tokenizer, device, config = load_model(checkpoint_path, config_path)
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return
    
    # Model info
    st.sidebar.subheader("Model Information")
    st.sidebar.write(f"**Device:** {device}")
    st.sidebar.write(f"**Video Encoder:** {config.model.video_encoder.model_name}")
    st.sidebar.write(f"**Text Decoder:** {config.model.text_decoder.model_name}")
    st.sidebar.write(f"**Fusion Type:** {config.model.fusion_type}")
    
    # Main content
    st.header("Upload Video")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv', 'wmv'],
        help="Upload a video file to generate captions"
    )
    
    if uploaded_file is not None:
        # Display video info
        st.subheader("Video Information")
        
        # Save video temporarily to get info
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        try:
            video_info = get_video_info(tmp_path)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Duration", f"{video_info['duration']:.1f}s")
            with col2:
                st.metric("FPS", f"{video_info['fps']:.1f}")
            with col3:
                st.metric("Resolution", f"{video_info['width']}x{video_info['height']}")
            with col4:
                st.metric("Frames", video_info['frame_count'])
            
        finally:
            os.unlink(tmp_path)
        
        # Reset file pointer
        uploaded_file.seek(0)
        
        # Generate caption button
        if st.button("Generate Caption", type="primary"):
            with st.spinner("Processing video..."):
                try:
                    # Process video
                    video_tensor, frames = process_video(uploaded_file, config)
                    
                    # Move to device
                    video_tensor = video_tensor.to(device)
                    
                    # Generate caption
                    with torch.no_grad():
                        captions = model.generate(
                            video_frames=video_tensor,
                            max_length=config.data.max_caption_length,
                            num_beams=5,
                            temperature=1.0,
                            do_sample=False,
                            early_stopping=True,
                        )
                    
                    caption = captions[0]
                    
                    # Display results
                    st.subheader("Generated Caption")
                    st.success(caption)
                    
                    # Display video frames
                    st.subheader("Video Frames")
                    
                    # Show frames in a grid
                    cols = st.columns(4)
                    for i, frame in enumerate(frames[:8]):  # Show first 8 frames
                        with cols[i % 4]:
                            st.image(frame, caption=f"Frame {i+1}")
                    
                    # Show additional frames if available
                    if len(frames) > 8:
                        st.info(f"Showing first 8 frames out of {len(frames)} total frames")
                    
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
    
    # Example section
    st.header("How to Use")
    st.markdown("""
    1. **Upload a video file** using the file uploader above
    2. **Click "Generate Caption"** to process the video
    3. **View the generated caption** and video frames
    
    The model will:
    - Extract frames from your video
    - Process them through a CLIP visual encoder
    - Generate a caption using a GPT-2 text decoder
    - Display the results
    """)
    
    # Technical details
    with st.expander("Technical Details"):
        st.markdown("""
        **Model Architecture:**
        - **Video Encoder:** CLIP ViT-B/32 for visual feature extraction
        - **Text Decoder:** GPT-2 for caption generation
        - **Fusion:** Cross-attention mechanism between visual and text features
        
        **Processing Pipeline:**
        1. Video frames are extracted at 1 FPS
        2. Frames are resized to 224x224 pixels
        3. Visual features are extracted using CLIP
        4. Captions are generated using GPT-2 with cross-attention
        5. Results are displayed with evaluation metrics
        
        **Supported Formats:** MP4, AVI, MOV, MKV, WMV
        **Max Video Length:** No limit (frames are sampled)
        **Output:** Natural language captions describing the video content
        """)


if __name__ == "__main__":
    main()
