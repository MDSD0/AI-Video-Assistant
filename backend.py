import os
import shutil
import base64
import logging
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
import tempfile
import google.generativeai as genai
import whisper
from TTS.api import TTS
from datetime import datetime
import re
from media_extractor import split_video

# Configure logging
logging.basicConfig(level=logging.INFO, filename="video_assistant.log")
logger = logging.getLogger(__name__)

class VideoAssistantBackend:
    def __init__(self, gemini_api_key: str):
        os.makedirs("frames", exist_ok=True)
        self.frame_dir = "frames"
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel("gemini-2.0-flash")
        self.whisper_model = whisper.load_model("small")
        self.tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False)
        self.response_cache = {}
        self.metrics_log = []

    def adaptive_frame_sampling(self, image_uris: List[str], threshold: float = 0.1) -> List[str]:
        if not image_uris:
            return []
        sampled_uris = [image_uris[0]]
        prev_frame = cv2.imread(image_uris[0], cv2.IMREAD_GRAYSCALE)

        for uri in image_uris[1:]:
            curr_frame = cv2.imread(uri, cv2.IMREAD_GRAYSCALE)
            if prev_frame is None or curr_frame is None or prev_frame.shape != curr_frame.shape:
                continue
            diff = np.mean((curr_frame - prev_frame) ** 2)
            if diff > threshold:
                sampled_uris.append(uri)
                prev_frame = curr_frame

        logger.info(f"Sampled {len(sampled_uris)} out of {len(image_uris)} frames.")
        return sampled_uris

    def transcribe_audio(self, audio_path: str) -> str:
        try:
            start_time = datetime.now()
            result = self.whisper_model.transcribe(audio_path)
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Transcription took {duration:.2f} seconds.")
            self.metrics_log.append({"task": "transcription", "duration": duration})
            return result["text"]
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

    def analyze_video(self, image_uris: List[str]) -> str:
        cache_key = "|".join(sorted(image_uris))
        if cache_key in self.response_cache:
            logger.info("Retrieved video analysis from cache.")
            return self.response_cache[cache_key]

        messages = [{"role": "user", "parts": [{"text": "Analyze the scene. Describe objects, actions, text, characters, and notable details."}]}]
        for image_uri in image_uris:
            try:
                with open(image_uri, "rb") as img_file:
                    image_data = img_file.read()
                messages.append({"role": "user", "parts": [{"inline_data": {"mime_type": "image/jpeg", "data": image_data}}]})
            except Exception as e:
                logger.error(f"Failed to process image {image_uri}: {e}")
                continue

        try:
            start_time = datetime.now()
            response = self.gemini_model.generate_content(contents=messages)
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Video analysis took {duration:.2f} seconds.")
            self.metrics_log.append({"task": "video_analysis", "duration": duration})
            self.response_cache[cache_key] = response.text
            return response.text
        except Exception as e:
            logger.error(f"Error in video analysis: {e}")
            return "Unable to analyze video content."

    def to_gemini_messages(self, history: List[Dict]) -> List[Dict]:
        gemini_messages = []
        for message in history:
            role = message["role"]
            parts = []
            for part in message["parts"]:
                if "text" in part:
                    parts.append({"text": part["text"]})
                elif "image_path" in part:
                    try:
                        with open(part["image_path"], "rb") as img_file:
                            image_data = img_file.read()
                        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": image_data}})
                    except Exception as e:
                        logger.error(f"Failed to load image {part['image_path']}: {e}")
                        continue
            gemini_messages.append({"role": role, "parts": parts})
        return gemini_messages
    def generate_response(self, history: List[Dict]) -> str:
        if not history:
            return "I do not have any past context to refer to."

        latest_text = history[-1]["parts"][0]["text"]
        latest_query = latest_text.lower()

        # Extract latest visual context if available
        latest_visual = None
        if len(history) >= 3:
            visual_entry = history[-3]
            if visual_entry["parts"][0]["text"].startswith("Visual context"):
                latest_visual = visual_entry["parts"][0]["text"].replace("Visual context: ", "")

        # Present tense → current video
        if re.search(r"\b(what am i|what is this|what is happening|who is|what are|do you see|what do you see|describe this|explain this)\b", latest_query) and latest_visual:
            return f"It looks like you're seeing: {latest_visual}"

        # Group videos into triplets: (visual, query, response)
        triplets = []
        for i in range(0, len(history) - 2, 3):
            v, q, r = history[i], history[i+1], history[i+2]
            if v["parts"][0]["text"].startswith("Visual context"):
                triplets.append((v, q, r))

        # Past tense queries: previous video
        if "previous video" in latest_query or "last video" in latest_query:
            if len(triplets) < 2:
                return "I do not have enough context to recall the previous video."
            v, q, r = triplets[-2]
            vtxt = v["parts"][0]["text"].replace("Visual context: ", "")
            qtxt = q["parts"][0]["text"].replace("User query: ", "")
            rtxt = r["parts"][0]["text"]
            if "did you see" in latest_query or "what did you see" in latest_query:
                return f"In the previous video, I saw: {vtxt}"
            if "did i ask" in latest_query or "what did i ask" in latest_query:
                return f"In the previous video, you asked: {qtxt}"
            if "did you say" in latest_query or "how did you respond" in latest_query or "what did you answer" in latest_query:
                return f"In the previous video, I said: {rtxt}"
            return f"Earlier, you asked: {qtxt}, and I responded: {rtxt}"

        # Specific video queries
        match = re.search(r"video (\d+)", latest_query)
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(triplets):
                v, q, r = triplets[idx]
                vtxt = v["parts"][0]["text"].replace("Visual context: ", "")
                qtxt = q["parts"][0]["text"].replace("User query: ", "")
                rtxt = r["parts"][0]["text"]
                if "did you see" in latest_query or "what did you see" in latest_query:
                    return f"In video {idx + 1}, I saw: {vtxt}"
                if "did i ask" in latest_query or "what did i ask" in latest_query:
                    return f"In video {idx + 1}, you asked: {qtxt}"
                return f"In video {idx + 1}, I said: {rtxt}"
            return f"I do not have context for video {idx + 1}"

        # Default: Ask Gemini
        weighted_history = self.apply_context_weights(history)
        gemini_messages = self.to_gemini_messages(weighted_history)
        gemini_messages.append({
            "role": "user",
            "parts": [{
                "text": (
                    "You are an AI Video Assistant. The user shows you a video and asks a question. "
                    "Answer based on the most recent visual context for present tense queries. "
                    "If asked about previous videos, retrieve from stored context. "
                    "Be clear, direct, and friendly in your responses. "
                    "Do NOT use symbols and Do NOT use '*' asterisks. Do not use markdown. Be conversational and natural."
                )
            }]
        })

        try:
            response = self.gemini_model.generate_content(contents=gemini_messages)
            return response.text
        except Exception:
            return "Sorry, I could not process that."

    def apply_context_weights(self, history: List[Dict]) -> List[Dict]:
        weighted_history = []
        decay_factor = 0.9
        for i, entry in enumerate(reversed(history)):
            weight = decay_factor ** i
            if weight < 0.1:
                continue
            weighted_history.append(entry)
        return list(reversed(weighted_history))

    def text_to_speech(self, text: str, output_path: str = "response.wav") -> str:
        try:
            start_time = datetime.now()
            self.tts.tts_to_file(text=text, speaker="p227", file_path=output_path)
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Text-to-speech synthesis took {duration:.2f} seconds.")
            self.metrics_log.append({"task": "text_to_speech", "duration": duration})
            return output_path
        except Exception as e:
            logger.error(f"Text-to-speech failed: {e}")
            raise

    def process_video(self, video_path: str, history: List[Dict], video_counter: int) -> Tuple[Optional[str], str, str, List[Dict], int]:
        if video_path is None:
            logger.warning("No video path provided.")
            return None, "", "", history or [], video_counter

        history = history or []
        video_counter += 1

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                audio_uri, image_uris = split_video(video_path, output_dir=temp_dir, prefix=f"video_{video_counter}")
            except Exception as e:
                logger.error(f"Error in split_video: {e}")
                return None, "", "", history, video_counter

            persistent_image_uris = []
            for i, image_uri in enumerate(image_uris):
                frame_path = os.path.join(self.frame_dir, f"frame_{video_counter}_{i}.jpg")
                try:
                    shutil.copy(image_uri, frame_path)
                    persistent_image_uris.append(frame_path)
                except Exception as e:
                    logger.error(f"Failed to save frame {image_uri}: {e}")
                    continue

            sampled_uris = self.adaptive_frame_sampling(persistent_image_uris)
            visual_context = self.analyze_video(sampled_uris)
            visual_message = {"role": "user", "parts": [{"text": f"Visual context: {visual_context}"}] + [{"image_path": path} for path in sampled_uris]}
            history.append(visual_message)

            user_prompt = self.transcribe_audio(audio_uri)
            query_message = {"role": "user", "parts": [{"text": f"User query: {user_prompt}"}]}
            history.append(query_message)

            response_text = self.generate_response(history)
            assistant_message = {"role": "assistant", "parts": [{"text": response_text}]}
            history.append(assistant_message)

            audio_path = self.text_to_speech(response_text)

        return audio_path, user_prompt, response_text, history, video_counter

    def get_metrics(self) -> List[Dict]:
        return self.metrics_log
