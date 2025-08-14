# AI Video Assistant

This project is an interactive AI assistant that analyzes video content. You can upload a short video and ask a question within the video's audio track. The assistant will watch the video, listen to your question, and provide a spoken answer based on what it sees.

## Core Features

- **Video and Audio Analysis**: Understands content from both visual frames and audio tracks.
- **Conversational Memory**: Remembers past videos and can answer questions about them.
- **Speech-to-Speech Interaction**: Takes a spoken question as input and provides a spoken audio response.
- **Efficient Processing**: Uses adaptive frame sampling to analyze only the most significant visual changes.
- **Simple Web Interface**: Easy-to-use interface for uploading videos and viewing results.

## How It Works

The application processes each video through a simple pipeline:

1. **Extract**: The video is split into image frames and an audio file.
2. **Analyze Vision**: Key visual frames are sent to the Google Gemini model to create a text description of the scene.
3. **Transcribe Audio**: The audio is converted to text using the OpenAI Whisper model to capture the user's question.
4. **Generate Response**: The visual description, the user's question, and past conversation history are used to generate a relevant text answer.
5. **Synthesize Speech**: The text answer is converted into an audio file, which is played back to the user.

## Tech Stack

- **Backend**: Python
- **AI Models**:
  - **Vision/Reasoning**: Google Gemini
  - **Speech-to-Text**: OpenAI Whisper
  - **Text-to-Speech**: Coqui TTS
- **Web Interface**: Gradio
- **Core Libraries**: OpenCV, NumPy, PyTorch

## Setup and Installation

Follow these instructions carefully to set up and run the project locally.

### Step 1: Install Prerequisites (FFmpeg)

This application requires **FFmpeg** for audio processing. You must install it and ensure it's available in your system's command line.

- **On Debian/Ubuntu Linux:**
  ```bash
  sudo apt update && sudo apt install ffmpeg
  ```

- **On macOS (using Homebrew):**
  ```bash
  brew install ffmpeg
  ```

- **On Windows:**
  1. Download the latest build from [ffmpeg.org/download.html](https://ffmpeg.org/download.html).
  2. Extract the downloaded archive (e.g., to `C:\ffmpeg`).
  3. Add the `bin` directory from the extracted folder (e.g., `C:\ffmpeg\bin`) to your system's **PATH** environment variable.

To verify the installation, open a new terminal or command prompt and run `ffmpeg -version`. You should see the version information.

### Step 2: Clone the Repository

Clone this repository to your local machine.

```bash
git clone https://github.com/MDSD0/AI-Video-Assistant.git
cd AI-Video-Assistant
```

### Step 3: Create a Python Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
# Create the environment
python -m venv venv

# Activate the environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 4: Install Dependencies

Install all the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### Step 5: Configure Your API Key

You will need a Gemini API Key from Google AI Studio. You can get one here: https://aistudio.google.com/app/apikey

In the project's root directory, create a new file named `.env`.

Add your API key to the `.env` file in the following format:

```
GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"
```

## Usage

Make sure your virtual environment is activated.

Run the `frontend.py` script to start the application.

```bash
python frontend.py
```

The terminal will display a local URL (e.g., http://127.0.0.1:7860). Open this URL in your web browser.

In the web interface, upload a video file. The video must contain your spoken question in its audio track.

Click the "Process Video" button.

The application will process the video, and the interface will update with the transcribed question, the AI's text response, and an audio player for the spoken response.
