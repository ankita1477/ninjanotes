from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import whisper
from transformers import pipeline
import os
import subprocess
from pathlib import Path
import numpy as np
import json
import time  # Add this import
import soundfile as sf  # Add this import for audio validation
import wave
import contextlib
from pydub import AudioSegment  # Add this import
from pytube import YouTube  # Add import
import yt_dlp  # Add this import for universal video downloading
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# FFmpeg configuration
FFMPEG_PATH = os.environ.get('FFMPEG_PATH', r"C:\ffmpeg-7.1-essentials_build\bin")
os.environ["PATH"] = FFMPEG_PATH + os.pathsep + os.environ["PATH"]

# Hugging Face configuration
HUGGINGFACE_API_KEY = os.environ.get('HUGGINGFACE_API_KEY', "your_default_key")
os.environ["HUGGINGFACE_TOKEN"] = HUGGINGFACE_API_KEY

app = Flask(__name__)

# Update configuration for Render deployment
UPLOAD_FOLDER = '/tmp/uploads' if os.environ.get('RENDER') else 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a', 'ogg'}

# Add global progress tracking
processing_progress = 0

# Add model status tracking
model_status = {
    'current_stage': None,
    'stages': {
        'input': {'status': 'waiting', 'details': ''},
        'whisper': {'status': 'waiting', 'details': ''},
        'bart': {'status': 'waiting', 'details': ''},
        'output': {'status': 'waiting', 'details': ''}
    }
}

def update_model_status(stage, status, details=''):
    """Update model processing status"""
    global model_status
    model_status['current_stage'] = stage
    model_status['stages'][stage] = {
        'status': status,
        'details': details
    }

def check_ffmpeg():
    try:
        ffmpeg_exe = os.path.join(FFMPEG_PATH, 'ffmpeg.exe')
        if not os.path.exists(ffmpeg_exe):
            return False
        subprocess.run([ffmpeg_exe, '-version'], capture_output=True, check=True)
        return True
    except:
        return False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize models
def initialize_models():
    try:
        transcriber = whisper.load_model("base")
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            token=HUGGINGFACE_API_KEY
        )
        return transcriber, summarizer
    except Exception as e:
        print(f"Model initialization error: {str(e)}")
        try:
            # Fallback to smaller models
            transcriber = whisper.load_model("tiny")
            summarizer = pipeline(
                "summarization",
                model="sshleifer/distilbart-cnn-12-6",
                token=HUGGINGFACE_API_KEY
            )
            return transcriber, summarizer
        except Exception as e2:
            print(f"Critical error: {str(e2)}")
            raise RuntimeError("Failed to initialize models")

# Initialize models
try:
    transcriber, summarizer = initialize_models()
except Exception as e:
    print("Fatal error: Could not initialize models. Exiting...")
    exit(1)

def get_status_message(progress):
    """Get status message based on progress"""
    if progress < 25:
        return "Analyzing audio file..."
    elif progress < 50:
        return "Converting speech to text..."
    elif progress < 75:
        return "Processing transcript..."
    elif progress < 90:
        return "Generating summary..."
    else:
        return "Finishing up..."

@app.route('/progress')
def progress():
    """Stream progress updates"""
    def generate():
        global processing_progress
        while processing_progress < 100:
            yield f"data: {json.dumps({'progress': processing_progress, 'message': get_status_message(processing_progress)})}\n\n"
            time.sleep(0.5)
        yield f"data: {json.dumps({'progress': 100, 'message': 'Complete!'})}\n\n"
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/model-status')
def get_model_status():
    """Get current model processing status"""
    return jsonify(model_status)

def validate_audio_file(file_path):
    """Enhanced audio file validation with multiple checks"""
    try:
        # Check if file exists and has size
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return False, "Empty or missing audio file"
            
        # Try reading with different methods based on file type
        file_ext = file_path.lower().split('.')[-1]
        
        if file_ext == 'wav':
            with contextlib.closing(wave.open(file_path, 'rb')) as wav_file:
                if wav_file.getnframes() == 0:
                    return False, "WAV file contains no frames"
                if wav_file.getsampwidth() == 0:
                    return False, "Invalid WAV file format"
        else:
            # For non-WAV files, use pydub
            try:
                audio = AudioSegment.from_file(file_path)
                if len(audio) == 0:
                    return False, "Audio file contains no data"
                if audio.duration_seconds < 0.1:  # Check minimum duration
                    return False, "Audio file is too short"
            except Exception as e:
                return False, f"Audio format error: {str(e)}"

        return True, "Audio file is valid"
        
    except Exception as e:
        return False, f"Audio validation error: {str(e)}"

@app.route('/process', methods=['POST'])
def process_audio():
    global processing_progress, model_status
    processing_progress = 0
    
    # Reset model status
    for stage in model_status['stages']:
        model_status['stages'][stage] = {'status': 'waiting', 'details': ''}
    
    filename = None
    
    try:
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No file uploaded'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'No file selected'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'status': 'error',
                'message': 'Unsupported file type'
            }), 400
        
        processing_progress = 10
        filename = Path(app.config['UPLOAD_FOLDER']) / file.filename
        filename = str(filename.resolve())
        file.save(filename)
        
        # Input validation stage
        update_model_status('input', 'processing', 'Validating audio file...')
        # Validate audio file
        is_valid, message = validate_audio_file(filename)
        if not is_valid:
            if os.path.exists(filename):
                os.remove(filename)
            return jsonify({
                'status': 'error',
                'message': f"Audio validation failed: {message}"
            }), 400

        # Try converting to WAV if not already WAV
        if not filename.lower().endswith('.wav'):
            try:
                audio = AudioSegment.from_file(filename)
                wav_filename = filename.rsplit('.', 1)[0] + '.wav'
                audio.export(wav_filename, format='wav')
                if os.path.exists(filename):
                    os.remove(filename)
                filename = wav_filename
            except Exception as e:
                raise ValueError(f"Audio conversion failed: {str(e)}")

        # Whisper model stage
        update_model_status('whisper', 'processing', 'Converting speech to text...')
        # Continue with transcription
        processing_progress = 30
        try:
            result = transcriber.transcribe(filename)
            if not result or not result.get("text"):
                raise ValueError("No speech detected in audio")
            transcript = result["text"].strip()
        except Exception as e:
            raise ValueError(f"Transcription failed: {str(e)}")

        processing_progress = 60
        # BART model stage
        update_model_status('bart', 'processing', 'Generating summary...')
        # Generate summary if transcript is long enough
        if len(transcript.split()) >= 10:
            try:
                max_length = min(130, max(30, len(transcript.split()) // 2))
                summary = summarizer(
                    transcript,
                    max_length=max_length,
                    min_length=min(10, max_length - 5)
                )[0]['summary_text']
            except Exception as e:
                summary = "Error generating summary. Using transcript only."
        else:
            summary = "Audio clip is too short to generate a meaningful summary."
        
        processing_progress = 90
        # Output stage
        update_model_status('output', 'complete', 'Processing complete')
        # Clean up
        os.remove(filename)
        
        processing_progress = 100
        return jsonify({
            'status': 'success',
            'transcript': transcript,
            'summary': summary
        })
        
    except Exception as e:
        processing_progress = 0
        # Update status on error
        current_stage = model_status['current_stage']
        if current_stage:
            update_model_status(current_stage, 'error', str(e))
        if filename and os.path.exists(filename):
            os.remove(filename)
        error_message = str(e)
        if "reshape tensor" in error_message:
            error_message = "Invalid audio format. Please ensure your file is a valid audio recording."
        return jsonify({
            'status': 'error',
            'message': error_message
        }), 500

@app.route('/download-video', methods=['POST'])
def download_video():
    global processing_progress, model_status
    data = request.get_json()
    video_url = data.get('video_url')
    if not video_url:
        return jsonify({'status': 'error', 'message': 'No video URL provided'}), 400
    
    processing_progress = 10
    update_model_status('input', 'processing', 'Downloading video...')
    
    try:
        # Create unique filename using timestamp
        timestamp = int(time.time())
        base_filename = f"video_{timestamp}"
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{base_filename}.mp4")
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{base_filename}.wav")
        
        # Configure yt-dlp
        ydl_opts = {
            'format': 'mp4',  # Specify mp4 format
            'outtmpl': video_path,
            'quiet': True,
            'no_warnings': True,
        }
        
        # Download video
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download([video_url])
            except Exception as e:
                raise Exception(f"Video download failed: {str(e)}")

        if not os.path.exists(video_path):
            raise FileNotFoundError("Video file not found after download")
            
        print(f"Video downloaded to: {video_path}")  # Debug log
        
        # Extract audio
        processing_progress = 30
        try:
            audio = AudioSegment.from_file(video_path)
            audio.export(audio_path, format="wav")
            os.remove(video_path)  # Remove video file after extraction
            
            if not os.path.exists(audio_path):
                raise FileNotFoundError("Audio file not found after extraction")
                
            print(f"Audio extracted to: {audio_path}")  # Debug log
        except Exception as e:
            if os.path.exists(video_path):
                os.remove(video_path)
            raise Exception(f"Audio extraction failed: {str(e)}")

        # Process the audio
        processing_progress = 40
        update_model_status('whisper', 'processing', 'Transcribing audio...')
        try:
            result = transcriber.transcribe(audio_path)
            transcript = result["text"].strip()
        except Exception as e:
            raise Exception(f"Transcription failed: {str(e)}")

        # Generate summary
        processing_progress = 70
        update_model_status('bart', 'processing', 'Generating summary...')
        if len(transcript.split()) >= 10:
            try:
                summary = summarizer(transcript, max_length=130, min_length=10)[0]['summary_text']
            except Exception as e:
                summary = "Error generating summary."
        else:
            summary = "Clip is too short for a meaningful summary."

        # Cleanup and return results
        processing_progress = 100
        update_model_status('output', 'complete', 'Processing complete')
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return jsonify({'status': 'success', 'transcript': transcript, 'summary': summary})

    except Exception as e:
        # Cleanup on error
        for path in [video_path, audio_path]:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except:
                pass
        
        print(f"Error processing video: {str(e)}")  # Debug log
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    if not check_ffmpeg():
        print("ERROR: FFmpeg is not installed. Please install FFmpeg:")
        print(f"Expected path: {os.path.join(FFMPEG_PATH, 'ffmpeg.exe')}")
        exit(1)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
