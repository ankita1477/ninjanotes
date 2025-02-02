from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import whisper
from transformers import pipeline, AutoModelForSeq2SeqGeneration, AutoTokenizer
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
from sklearn.cluster import AgglomerativeClustering
from pyannote.audio import Pipeline
import datetime

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

# Initialize additional models
try:
    # Speaker diarization model
    diarization = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=HUGGINGFACE_API_KEY
    )
    
    # Key points extraction model
    key_points_model = AutoModelForSeq2SeqGeneration.from_pretrained("facebook/bart-large-cnn")
    key_points_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
except Exception as e:
    print(f"Warning: Could not load some models: {e}")

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

def analyze_transcript(transcript, audio_length):
    """Enhanced transcript analysis"""
    try:
        # Extract key points
        inputs = key_points_tokenizer.encode(
            "extract key points: " + transcript,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        )
        key_points = key_points_model.generate(
            inputs,
            max_length=150,
            min_length=40,
            num_beams=4,
            length_penalty=2.0
        )
        key_points = key_points_tokenizer.decode(key_points[0])
        
        # Basic sentiment analysis
        sentiment_analyzer = pipeline("sentiment-analysis")
        sentiment = sentiment_analyzer(transcript[:512])[0]
        
        # Calculate speech metrics
        words = transcript.split()
        word_count = len(words)
        speaking_rate = word_count / (audio_length / 60)  # words per minute
        
        return {
            "key_points": key_points,
            "sentiment": sentiment,
            "metrics": {
                "word_count": word_count,
                "duration_minutes": round(audio_length / 60, 2),
                "speaking_rate": round(speaking_rate, 1)
            }
        }
    except Exception as e:
        print(f"Analysis error: {e}")
        return None

def process_with_timestamps(audio_path):
    """Process audio with speaker detection and timestamps"""
    try:
        # Get audio duration
        audio = AudioSegment.from_file(audio_path)
        duration = len(audio) / 1000  # duration in seconds
        
        # Perform speaker diarization
        diarization_result = diarization(audio_path)
        
        # Transcribe with timestamps
        result = transcriber.transcribe(
            audio_path,
            task="transcribe",
            return_segments=True
        )
        
        # Combine speaker detection with transcription
        segments = []
        for segment in result["segments"]:
            start_time = str(datetime.timedelta(seconds=int(segment["start"])))
            end_time = str(datetime.timedelta(seconds=int(segment["end"])))
            
            # Find speaker for this segment
            speaker = "Speaker Unknown"
            for turn, _, speaker_id in diarization_result.itertracks(yield_label=True):
                if turn.start <= segment["start"] <= turn.end:
                    speaker = f"Speaker {speaker_id}"
                    break
            
            segments.append({
                "start": start_time,
                "end": end_time,
                "speaker": speaker,
                "text": segment["text"]
            })
        
        # Generate full transcript and analysis
        full_transcript = " ".join(s["text"] for s in segments)
        analysis = analyze_transcript(full_transcript, duration)
        
        return {
            "segments": segments,
            "transcript": full_transcript,
            "analysis": analysis
        }
        
    except Exception as e:
        print(f"Processing error: {e}")
        return None

@app.route('/download-video', methods=['POST'])
def download_video():
    global processing_progress, model_status
    data = request.get_json()
    video_url = data.get('video_url')
    if not video_url:
        return jsonify({'status': 'error', 'message': 'No video URL provided'}), 400
    
    processing_progress = 5
    update_model_status('input', 'processing', 'Starting video download...')
    
    try:
        # Create unique filename using timestamp
        timestamp = int(time.time())
        base_filename = f"video_{timestamp}"
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{base_filename}.mp4")
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{base_filename}.wav")
        
        def progress_hook(d):
            if d['status'] == 'downloading':
                progress = float(d['downloaded_bytes'] / d['total_bytes']) * 20 if d['total_bytes'] else 0
                processing_progress = 5 + progress
                update_model_status('input', 'processing', f"Downloading: {d.get('_percent_str', '0%')}")
            elif d['status'] == 'finished':
                processing_progress = 25
                update_model_status('input', 'complete', 'Download complete')
        
        # Configure yt-dlp with progress hook
        ydl_opts = {
            'format': 'bestaudio/best',  # Changed to prefer audio formats
            'extract_audio': True,  # Extract audio
            'audio_format': 'wav',  # Convert to WAV
            'outtmpl': video_path,
            'quiet': True,
            'no_warnings': True,
            'progress_hooks': [progress_hook],
        }
        
        # Download video/audio with progress tracking
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(video_url, download=True)
                if not info:
                    raise Exception("Failed to extract video information")
                downloaded_file = ydl.prepare_filename(info)
            except Exception as e:
                raise Exception(f"Video download failed: {str(e)}")

        if not os.path.exists(downloaded_file):
            raise FileNotFoundError("Downloaded file not found")
            
        print(f"File downloaded to: {downloaded_file}")
        
        # Convert to WAV if needed
        processing_progress = 30
        update_model_status('whisper', 'processing', 'Processing audio...')
        try:
            # Load and validate audio
            audio = AudioSegment.from_file(downloaded_file)
            if len(audio) == 0:
                raise ValueError("Audio file is empty")
            if audio.duration_seconds < 0.1:
                raise ValueError("Audio is too short")
                
            # Export as WAV with specific parameters
            audio = audio.set_channels(1)  # Convert to mono
            audio = audio.set_frame_rate(16000)  # Set sample rate to 16kHz
            audio.export(audio_path, format="wav", parameters=["-ac", "1", "-ar", "16000"])
            
            if os.path.exists(downloaded_file) and downloaded_file != audio_path:
                os.remove(downloaded_file)
            
            if not os.path.exists(audio_path):
                raise FileNotFoundError("WAV conversion failed")
                
            print(f"Audio processed and saved to: {audio_path}")
            processing_progress = 40
        except Exception as e:
            if os.path.exists(downloaded_file):
                os.remove(downloaded_file)
            raise Exception(f"Audio processing failed: {str(e)}")

        # Validate WAV file before transcription
        try:
            with wave.open(audio_path, 'rb') as wav_file:
                if wav_file.getnframes() == 0:
                    raise ValueError("WAV file contains no frames")
                if wav_file.getsampwidth() == 0:
                    raise ValueError("Invalid WAV format")
        except Exception as e:
            raise ValueError(f"Invalid WAV file: {str(e)}")

        # Process audio with enhanced features
        processing_progress = 50
        update_model_status('whisper', 'processing', 'Processing audio with speaker detection...')
        
        result = process_with_timestamps(audio_path)
        if not result:
            raise ValueError("Failed to process audio")
            
        processing_progress = 70
        update_model_status('bart', 'processing', 'Generating analysis...')
        
        # Generate final response
        response = {
            'status': 'success',
            'transcript': result['transcript'],
            'segments': result['segments'],
            'analysis': result['analysis'],
            'summary': summarizer(result['transcript'], max_length=130, min_length=10)[0]['summary_text']
        }
        
        # Cleanup and return results
        processing_progress = 100
        update_model_status('output', 'complete', 'Processing complete')
        if os.path.exists(audio_path):
            os.remove(audio_path)
            
        return jsonify(response)
        
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
