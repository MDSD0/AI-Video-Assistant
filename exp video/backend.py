import os
import shutil
import base64
import logging
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
from media_extractor import split_video
import google.generativeai as genai
import whisper
from TTS.api import TTS
from datetime import datetime
import psutil
import json
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO, filename="video_assistant.log")
logger = logging.getLogger(__name__)

class VideoAssistantBackend:
    def __init__(self, gemini_api_key: str):
        """Initialize models and setup directories."""
        self.temp_dir = tempfile.mkdtemp(prefix="video_assistant_")
        self.frame_dir = os.path.join(self.temp_dir, "frames")
        os.makedirs(self.frame_dir, exist_ok=True)
        
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        self.whisper_model = whisper.load_model("small")
        self.tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False)
        self.response_cache = {}  # Cache for Gemini responses
        self.cache_hits = 0
        self.cache_misses = 0
        self.metrics_log = []
        self.process = psutil.Process()

    def __del__(self):
        """Clean up temporary files when the object is destroyed."""
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.warning(f"Failed to clean up temp directory: {e}")

    def adaptive_frame_sampling(self, image_uris: List[str], threshold: float = 0.1) -> List[str]:
        """Sample frames adaptively based on scene changes."""
        if not image_uris:
            return []
        
        sampled_uris = [image_uris[0]]
        prev_frame = cv2.imread(image_uris[0], cv2.IMREAD_GRAYSCALE)

        for uri in image_uris[1:]:
            try:
                curr_frame = cv2.imread(uri, cv2.IMREAD_GRAYSCALE)
                if prev_frame.shape != curr_frame.shape:
                    continue
                diff = np.mean((curr_frame - prev_frame) ** 2)
                if diff > threshold:
                    sampled_uris.append(uri)
                    prev_frame = curr_frame
            except Exception as e:
                logger.warning(f"Error processing frame {uri}: {e}")
                continue

        logger.info(f"Sampled {len(sampled_uris)} out of {len(image_uris)} frames.")
        self.metrics_log.append({
            "task": "frame_sampling",
            "total_frames": len(image_uris),
            "sampled_frames": len(sampled_uris),
            "sampling_ratio": len(sampled_uris) / len(image_uris) if image_uris else 0
        })
        return sampled_uris

    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio using Whisper."""
        try:
            start_time = datetime.now()
            result = self.whisper_model.transcribe(audio_path)
            duration = (datetime.now() - start_time).total_seconds()
            transcription = result["text"]
            
            logger.info(f"Transcription took {duration:.2f} seconds.")
            self.metrics_log.append({
                "task": "transcription",
                "duration": duration,
                "memory_usage_mb": self.process.memory_info().rss / 1024 / 1024
            })
            return transcription
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return "Could not transcribe audio."

    def analyze_video(self, image_uris: List[str]) -> str:
        """Analyze video frames with Gemini, using cache for efficiency."""
        cache_key = "|".join(sorted(image_uris))
        if cache_key in self.response_cache:
            self.cache_hits += 1
            logger.info("Retrieved video analysis from cache.")
            return self.response_cache[cache_key]
        self.cache_misses += 1

        messages = [{
            "role": "user",
            "parts": [{"text": "Analyze the scene. Describe objects, actions, text, characters, and notable details."}]
        }]
        
        for image_uri in image_uris:
            try:
                with open(image_uri, "rb") as img_file:
                    image_data = img_file.read()
                messages.append({
                    "role": "user",
                    "parts": [{"inline_data": {"mime_type": "image/jpeg", "data": image_data}}]
                })
            except Exception as e:
                logger.error(f"Failed to process image {image_uri}: {e}")
                continue

        try:
            start_time = datetime.now()
            response = self.gemini_model.generate_content(contents=messages)
            duration = (datetime.now() - start_time).total_seconds()
            analysis = response.text
            
            logger.info(f"Video analysis took {duration:.2f} seconds.")
            self.metrics_log.append({
                "task": "video_analysis",
                "duration": duration,
                "memory_usage_mb": self.process.memory_info().rss / 1024 / 1024,
                "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
            })
            
            self.response_cache[cache_key] = analysis
            return analysis
        except Exception as e:
            logger.error(f"Error in video analysis: {e}")
            return "Unable to analyze video content."

    def generate_response(self, history: List[Dict]) -> str:
        """Generate a context-aware response using Gemini."""
        try:
            start_time = datetime.now()
            response = self.gemini_model.generate_content(contents=history)
            duration = (datetime.now() - start_time).total_seconds()
            response_text = response.text
            
            logger.info(f"Response generation took {duration:.2f} seconds.")
            self.metrics_log.append({
                "task": "response_generation",
                "duration": duration,
                "memory_usage_mb": self.process.memory_info().rss / 1024 / 1024
            })
            return response_text
        except Exception as e:
            logger.error(f"Error in Gemini API call: {e}")
            return "Sorry, I couldn't process that!"

    def text_to_speech(self, text: str) -> str:
        """Convert text to speech and return the audio file path."""
        try:
            start_time = datetime.now()
            output_path = os.path.join(self.temp_dir, "response.wav")
            self.tts.tts_to_file(text=text, speaker="p227", file_path=output_path)
            duration = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Text-to-speech synthesis took {duration:.2f} seconds.")
            self.metrics_log.append({
                "task": "text_to_speech",
                "duration": duration,
                "memory_usage_mb": self.process.memory_info().rss / 1024 / 1024
            })
            return output_path
        except Exception as e:
            logger.error(f"Text-to-speech failed: {e}")
            raise

    def process_video(self, video_path: str, history: List[Dict], video_counter: int) -> Tuple[str, str, str, List[Dict], int]:
        """Main processing pipeline for video input."""
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return None, "", "", history or [], video_counter

        history = history or []
        video_counter += 1
        start_time_total = datetime.now()

        try:
            # Extract audio and frames
            audio_uri, image_uris = split_video(video_path, self.frame_dir, f"vid_{video_counter}")
            
            # Save frames persistently
            persistent_image_uris = []
            for i, image_uri in enumerate(image_uris):
                frame_path = os.path.join(self.frame_dir, f"frame_{video_counter}_{i}.jpg")
                shutil.copy(image_uri, frame_path)
                persistent_image_uris.append(frame_path)

            # Apply adaptive frame sampling
            sampled_uris = self.adaptive_frame_sampling(persistent_image_uris)
            
            # Analyze video content
            visual_context = self.analyze_video(sampled_uris)
            visual_message = {
                "role": "user",
                "parts": [{"text": f"Visual context: {visual_context}"}] + 
                         [{"image_path": path} for path in sampled_uris]
            }
            history.append(visual_message)

            # Transcribe audio for user query
            user_prompt = self.transcribe_audio(audio_uri)
            query_message = {
                "role": "user",
                "parts": [{"text": f"User query: {user_prompt}"}]
            }
            history.append(query_message)

            # Generate response
            response_text = self.generate_response(history)
            
            # Add assistant response to history
            assistant_message = {"role": "assistant", "parts": [{"text": response_text}]}
            history.append(assistant_message)

            # Generate audio
            audio_path = self.text_to_speech(response_text)

            # Log total processing time
            total_duration = (datetime.now() - start_time_total).total_seconds()
            self.metrics_log.append({
                "task": "end_to_end",
                "duration": total_duration,
                "video_length": len(image_uris),
                "memory_usage_mb": self.process.memory_info().rss / 1024 / 1024
            })

            return audio_path, user_prompt, response_text, history, video_counter

        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return None, "", "An error occurred while processing the video.", history, video_counter

    def get_metrics(self) -> List[Dict]:
        """Return logged performance metrics."""
        return self.metrics_log