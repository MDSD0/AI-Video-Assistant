from flask import Flask, render_template, request, jsonify
from backend import VideoAssistantBackend
import dotenv
import os
import uuid
from werkzeug.utils import secure_filename
import time

# Load environment variables
dotenv.load_dotenv()
if not os.environ.get("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY not found in .env file")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB limit
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize backend
backend = VideoAssistantBackend(gemini_api_key=os.environ["GEMINI_API_KEY"])

# Store session data (in production use Redis or database)
sessions = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    session_id = str(uuid.uuid4())
    filename = secure_filename(f"{session_id}_{file.filename}")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Initialize session
    sessions[session_id] = {
        'video_path': filepath,
        'history': [],
        'video_counter': 0,
        'created_at': time.time()
    }
    
    return jsonify({
        'session_id': session_id,
        'video_url': f'/static/uploads/{filename}'
    })

@app.route('/process', methods=['POST'])
def process_video():
    data = request.json
    session_id = data.get('session_id')
    
    if not session_id or session_id not in sessions:
        return jsonify({'error': 'Invalid session'}), 400
    
    session = sessions[session_id]
    
    # Process video
    try:
        audio_path, transcription, response, history, counter = backend.process_video(
            session['video_path'],
            session['history'],
            session['video_counter']
        )
        
        # Update session
        session['history'] = history
        session['video_counter'] = counter
        
        return jsonify({
            'audio_url': f'/{audio_path}',
            'transcription': transcription,
            'response': response
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)