"""Video processing utilities."""

import cv2
import numpy as np
import torch
from PIL import Image
from typing import List, Optional, Tuple, Union


def extract_frames_from_video(
    video_path: str,
    fps: float = 1.0,
    max_frames: Optional[int] = None,
    frame_size: Optional[Tuple[int, int]] = None,
) -> List[np.ndarray]:
    """Extract frames from a video file.
    
    Args:
        video_path: Path to the video file.
        fps: Frames per second to extract.
        max_frames: Maximum number of frames to extract.
        frame_size: Target frame size (height, width).
        
    Returns:
        List of extracted frames as numpy arrays.
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate frame interval
    frame_interval = max(1, int(video_fps / fps))
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Extract frame at specified intervals
        if frame_count % frame_interval == 0:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize if specified
            if frame_size is not None:
                frame = cv2.resize(frame, (frame_size[1], frame_size[0]))
            
            frames.append(frame)
            
            # Check if we've reached max_frames
            if max_frames is not None and len(frames) >= max_frames:
                break
        
        frame_count += 1
    
    cap.release()
    return frames


def frames_to_tensor(frames: List[np.ndarray]) -> torch.Tensor:
    """Convert list of frames to PyTorch tensor.
    
    Args:
        frames: List of frames as numpy arrays.
        
    Returns:
        PyTorch tensor of shape (T, H, W, C).
    """
    if not frames:
        raise ValueError("No frames provided")
    
    # Stack frames and convert to tensor
    frames_array = np.stack(frames, axis=0)
    tensor = torch.from_numpy(frames_array).float()
    
    # Normalize to [0, 1]
    tensor = tensor / 255.0
    
    return tensor


def tensor_to_frames(tensor: torch.Tensor) -> List[np.ndarray]:
    """Convert PyTorch tensor to list of frames.
    
    Args:
        tensor: PyTorch tensor of shape (T, H, W, C) or (T, C, H, W).
        
    Returns:
        List of frames as numpy arrays.
    """
    # Convert to numpy
    if tensor.dim() == 4:
        if tensor.shape[1] == 3:  # (T, C, H, W)
            tensor = tensor.permute(0, 2, 3, 1)  # (T, H, W, C)
        
        # Denormalize
        tensor = tensor * 255.0
        tensor = torch.clamp(tensor, 0, 255)
        
        frames = tensor.numpy().astype(np.uint8)
        return [frames[i] for i in range(frames.shape[0])]
    else:
        raise ValueError(f"Expected 4D tensor, got {tensor.dim()}D")


def resize_frames(
    frames: List[np.ndarray],
    size: Tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR,
) -> List[np.ndarray]:
    """Resize frames to specified size.
    
    Args:
        frames: List of frames as numpy arrays.
        size: Target size (height, width).
        interpolation: OpenCV interpolation method.
        
    Returns:
        List of resized frames.
    """
    resized_frames = []
    for frame in frames:
        resized_frame = cv2.resize(frame, (size[1], size[0]), interpolation=interpolation)
        resized_frames.append(resized_frame)
    
    return resized_frames


def pad_frames(
    frames: List[np.ndarray],
    target_length: int,
    pad_value: int = 0,
) -> List[np.ndarray]:
    """Pad frames to target length.
    
    Args:
        frames: List of frames as numpy arrays.
        target_length: Target number of frames.
        pad_value: Value to use for padding.
        
    Returns:
        List of padded frames.
    """
    if len(frames) >= target_length:
        return frames[:target_length]
    
    # Create padding frame
    if frames:
        pad_frame = np.full_like(frames[0], pad_value)
    else:
        raise ValueError("Cannot pad empty frame list")
    
    # Pad with repeated frames
    padded_frames = frames.copy()
    while len(padded_frames) < target_length:
        padded_frames.append(pad_frame)
    
    return padded_frames


def create_video_from_frames(
    frames: List[np.ndarray],
    output_path: str,
    fps: float = 30.0,
    codec: str = "mp4v",
) -> None:
    """Create video from list of frames.
    
    Args:
        frames: List of frames as numpy arrays.
        output_path: Output video path.
        fps: Frames per second.
        codec: Video codec.
    """
    if not frames:
        raise ValueError("No frames provided")
    
    height, width = frames[0].shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()


def get_video_info(video_path: str) -> dict:
    """Get video information.
    
    Args:
        video_path: Path to the video file.
        
    Returns:
        Dictionary containing video information.
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
    }
    
    cap.release()
    return info
