import os
import cv2
import subprocess
import tempfile
from typing import Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def split_video(video_path: str, output_dir: str, prefix: str = "frame") -> Tuple[str, List[str]]:
    """
    Extract audio and frames from video file.
    Returns tuple of (audio_path, frame_paths)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract audio using FFmpeg
    audio_path = os.path.join(output_dir, f"{prefix}_audio.wav")
    try:
        # Add input format probing to handle various formats (e.g., WebM)
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path,
            "-ac", "1", "-ar", "16000",  # Convert to mono 16kHz
            "-vn", "-f", "wav", audio_path
        ], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg audio extraction failed: {e.stderr}")
        raise RuntimeError(f"Audio extraction failed: {e.stderr}")
    
    # Extract frames using OpenCV
    frame_paths = []
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(fps / 3))  # Sample 3 frames per second
    
    success, image = vidcap.read()
    count = 0
    while success:
        if count % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"{prefix}_frame{count}.jpg")
            cv2.imwrite(frame_path, image)
            frame_paths.append(frame_path)
        success, image = vidcap.read()
        count += 1
    
    vidcap.release()
    
    if not frame_paths:
        logger.error(f"No frames extracted from video: {video_path}")
        raise RuntimeError("No frames extracted from video")
    
    logger.info(f"Extracted {len(frame_paths)} frames and audio from {video_path}")
    return audio_path, frame_paths