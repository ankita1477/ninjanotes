from flask import Flask, render_template, request, jsonify, session, url_for, redirect, Response, stream_with_context
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
import whisper
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import os
import subprocess
from pathlib import Path
import pyaudio
import wave
import threading
import time
from playwright.sync_api import sync_playwright
import sounddevice as sd
import soundfile as sf
import numpy as np
import json
import nltk
from nltk.tokenize import sent_tokenize
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Direct API key configuration
HUGGINGFACE_API_KEY = "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"
os.environ["HUGGINGFACE_TOKEN"] = HUGGINGFACE_API_KEY

# Update FFmpeg path to essentials build bin directory
FFMPEG_PATH = r"C:\ffmpeg-2025-01-30-git-1911a6ec26-essentials_build\bin"
os.environ["PATH"] = FFMPEG_PATH + os.pathsep + os.environ["PATH"]

def check_ffmpeg():
    try:
        # Check for ffmpeg.exe directly in the directory
        ffmpeg_exe = os.path.join(FFMPEG_PATH, 'ffmpeg.exe')
        if not os.path.exists(ffmpeg_exe):
            print(f"FFmpeg executable not found at: {ffmpeg_exe}")
            return False
        result = subprocess.run([ffmpeg_exe, '-version'], capture_output=True, check=True, text=True)
        print(f"FFmpeg found successfully at: {ffmpeg_exe}")
        print(f"FFmpeg version info: {result.stdout.splitlines()[0]}")
        return True
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        print(f"Error checking FFmpeg: {str(e)}")
        return False

app = Flask(__name__)
# Add secret key for session management
app.secret_key = os.urandom(24)
# Update upload folder path
app.config['UPLOAD_FOLDER'] = r'F:\vercal\ninjanotes\uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a', 'ogg'}

# Ensure upload directory exists with proper path
Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)

# Check FFmpeg installation
if not check_ffmpeg():
    print("ERROR: FFmpeg is not installed. Please install FFmpeg first.")
    exit(1)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize models with API token and error handling
def initialize_models():
    try:
        print("Initializing NLP models with direct API key...")
        
        # Initialize models with direct API key
        transcriber = whisper.load_model("base")
        print("✓ Whisper model loaded")
        
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            token=HUGGINGFACE_API_KEY
        )
        print("✓ Summarization model loaded")
        
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            token=HUGGINGFACE_API_KEY
        )
        print("✓ Sentiment analysis model loaded")
        
        zero_shot_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            token=HUGGINGFACE_API_KEY
        )
        print("✓ Zero-shot classifier loaded")
        
        return transcriber, summarizer, sentiment_analyzer, zero_shot_classifier
        
    except Exception as e:
        print(f"Error during model initialization: {str(e)}")
        print("Attempting fallback initialization...")
        
        try:
            # Fallback to simpler models with direct API key
            transcriber = whisper.load_model("tiny")
            summarizer = pipeline("summarization", 
                                model="sshleifer/distilbart-cnn-12-6",
                                token=HUGGINGFACE_API_KEY)
            sentiment_analyzer = pipeline("sentiment-analysis",
                                       token=HUGGINGFACE_API_KEY)
            zero_shot_classifier = pipeline("zero-shot-classification",
                                         model="valhalla/distilbart-mnli-12-3",
                                         token=HUGGINGFACE_API_KEY)
            print("✓ Fallback models loaded successfully")
            return transcriber, summarizer, sentiment_analyzer, zero_shot_classifier
            
        except Exception as e2:
            print(f"Critical error: Could not initialize models: {str(e2)}")
            raise RuntimeError("Failed to initialize required models")

# Initialize models
try:
    transcriber, summarizer, sentiment_analyzer, zero_shot_classifier = initialize_models()
except Exception as e:
    print("Fatal error: Could not initialize models. Exiting...")
    exit(1)

# Google Meet configuration
SCOPES = ['https://www.googleapis.com/auth/meets.recordings']
CLIENT_SECRETS_FILE = "client_secrets.json"

class SimpleMeetRecorder:
    def __init__(self):
        self.is_recording = False
        self.playwright = None
        self.browser = None
        self.page = None
        
    def join_meet(self, meet_link):
        try:
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(
                headless=False,
                args=[
                    '--use-fake-ui-for-media-stream',
                    '--allow-file-access-from-files',
                    '--use-fake-device-for-media-stream'
                ]
            )
            
            context = self.browser.new_context(
                permissions=['camera', 'microphone']
            )
            
            self.page = context.new_page()
            self.page.goto(meet_link)
            
            # Handle initial dialogs (if any)
            self.page.click('text=Dismiss')
            
            # Turn off camera and microphone
            self.page.click('[aria-label="Turn off camera (ctrl + e)"]')
            self.page.click('[aria-label="Turn off microphone (ctrl + d)"]')
            
            # Join meeting
            self.page.click('text=Join now')
            print("Joined meeting successfully")
            return True
            
        except Exception as e:
            print(f"Error joining meet: {str(e)}")
            self.cleanup()
            return False
            
    def start_recording(self):
        self.is_recording = True
        threading.Thread(target=self._record_audio).start()
        
    def _record_audio(self):
        samplerate = 44100
        channels = 2
        filename = f"meet_recording_{int(time.time())}.wav"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Initialize recording
        with sd.InputStream(samplerate=samplerate, channels=channels) as stream:
            frames = []
            while self.is_recording:
                data, overflowed = stream.read(1024)
                frames.append(data)
                
        # Save and process recording
        if frames:
            audio_data = np.concatenate(frames, axis=0)
            sf.write(filepath, audio_data, samplerate)
            self.process_recording(filename)
    
    def stop_recording(self):
        self.is_recording = False
        self.cleanup()
        
    def cleanup(self):
        if self.page:
            self.page.close()
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()
            
        self.page = None
        self.browser = None
        self.playwright = None

    def process_recording(self, audio_file):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_file)
        result = transcriber.transcribe(file_path)
        transcript = result["text"]
        summary = summarizer(transcript, max_length=130, min_length=30)[0]['summary_text']
        
        # Save results
        with open(f'meet_summary_{int(time.time())}.txt', 'w') as f:
            f.write(f"Transcript:\n{transcript}\n\nSummary:\n{summary}")
        
        os.remove(file_path)

class EnhancedProcessor:
    def __init__(self):
        nltk.download('punkt', quiet=True)
        self.action_keywords = [
            "todo", "action item", "follow up", "task", "deadline", 
            "assign", "responsibility", "needs to", "should", "must",
            "required", "action required", "deliverable"
        ]
        self.meeting_categories = [
            "Project Update", "Planning", "Review", "Brainstorming", 
            "Status Report", "Decision Making", "Training", "Client Meeting",
            "Team Sync", "Technical Discussion"
        ]

    def analyze_text(self, transcript):
        try:
            if not transcript or len(transcript.split()) < 10:
                return self._get_default_analysis()

            sentences = sent_tokenize(transcript)
            
            # Analyze sentiment with retry mechanism
            sentiment = self._retry_operation(
                lambda: sentiment_analyzer(transcript[:512])[0],
                default={"label": "NEUTRAL", "score": 0.5}
            )
            
            # Extract action items with improved detection
            action_items = self._extract_action_items(sentences)
            
            # Identify speakers with better pattern matching
            speakers = self._identify_speakers(transcript)
            
            # Only classify if we have enough text
            if len(transcript.split()) >= 20:
                meeting_type = self._retry_operation(
                    lambda: self._classify_meeting_type(transcript),
                    default={"type": "General Meeting", "confidence": 0.0}
                )
            else:
                meeting_type = {"type": "Short Meeting", "confidence": 1.0}
            
            # Generate key points
            key_points = self._extract_key_points(sentences)
            
            return {
                "sentiment": {
                    "label": sentiment["label"],
                    "score": float(sentiment["score"])
                },
                "action_items": action_items,
                "speakers": speakers,
                "meeting_type": meeting_type,
                "key_points": key_points
            }
        except Exception as e:
            print(f"Analysis error: {str(e)}")
            return self._get_default_analysis()

    def _retry_operation(self, operation, max_retries=3, default=None):
        for attempt in range(max_retries):
            try:
                return operation()
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    return default
                time.sleep(1)  # Wait before retry

    def _get_default_analysis(self):
        return {
            "sentiment": {"label": "NEUTRAL", "score": 0.5},
            "action_items": [],
            "speakers": [],
            "meeting_type": {"type": "Unknown", "confidence": 0.0},
            "key_points": []
        }

    def _extract_action_items(self, sentences):
        action_items = []
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in self.action_keywords):
                action_items.append(sentence.strip())
        return action_items

    def _identify_speakers(self, transcript):
        # Basic speaker identification using common patterns
        speakers = set()
        speaker_patterns = [": ", "] ", "> "]
        
        for line in transcript.split('\n'):
            for pattern in speaker_patterns:
                if pattern in line:
                    potential_speaker = line.split(pattern)[0].strip()
                    if len(potential_speaker) < 50:  # Reasonable name length
                        speakers.add(potential_speaker)
        
        return list(speakers)

    def _classify_meeting_type(self, transcript):
        if not transcript or len(transcript.split()) < 20:
            return {"type": "Short Meeting", "confidence": 1.0}
            
        result = zero_shot_classifier(
            transcript[:512],
            candidate_labels=self.meeting_categories,
            hypothesis_template="This is a {}"
        )
        return {
            "type": result["labels"][0],
            "confidence": float(result["scores"][0])
        }

    def _extract_key_points(self, sentences):
        return [s.strip() for s in sentences if len(s.split()) > 5][:5]  # Top 5 substantial sentences

class ProcessingManager:
    @staticmethod
    def calculate_max_length(text):
        """Dynamically calculate max_length based on input length"""
        input_length = len(text.split())
        if (input_length < 10):
            return input_length
        return min(130, max(30, input_length // 2))

    @staticmethod
    def is_valid_transcript(text):
        """Check if transcript is valid for processing"""
        return bool(text and len(text.split()) >= 3)

# Create processor instance
enhanced_processor = EnhancedProcessor()

meet_recorder = SimpleMeetRecorder()

# Add this global variable for progress tracking
processing_progress = 0

def generate_progress():
    global processing_progress
    while processing_progress < 100:
        time.sleep(0.5)  # Update every 500ms
        yield f"data: {json.dumps({'progress': processing_progress, 'status': get_status_message(processing_progress)})}\n\n"
    yield f"data: {json.dumps({'progress': 100, 'status': 'Processing complete!'})}\n\n"

def get_status_message(progress):
    if progress < 30:
        return "Loading audio file..."
    elif progress < 60:
        return "Transcribing audio..."
    elif progress < 90:
        return "Generating summary..."
    else:
        return "Finalizing results..."

@app.route('/progress')
def progress():
    return Response(stream_with_context(generate_progress()), 
                   mimetype='text/event-stream')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/auth/google')
def google_auth():
    flow = Flow.from_client_secrets_file(CLIENT_SECRETS_FILE, scopes=SCOPES)
    flow.redirect_uri = url_for('oauth2callback', _external=True)
    authorization_url, state = flow.authorization_url(access_type='offline')
    session['state'] = state
    return redirect(authorization_url)

@app.route('/oauth2callback')
def oauth2callback():
    state = session['state']
    flow = Flow.from_client_secrets_file(CLIENT_SECRETS_FILE, scopes=SCOPES, state=state)
    flow.redirect_uri = url_for('oauth2callback', _external=True)
    authorization_response = request.url
    flow.fetch_token(authorization_response=authorization_response)
    credentials = flow.credentials
    session['credentials'] = {
        'token': credentials.token,
        'refresh_token': credentials.refresh_token,
        'token_uri': credentials.token_uri,
        'client_id': credentials.client_id,
        'client_secret': credentials.client_secret,
        'scopes': credentials.scopes
    }
    return redirect(url_for('index'))

@app.route('/join-meet', methods=['POST'])
def join_meet():
    meet_link = request.json.get('meet_link')
    if not meet_link:
        return jsonify({'error': 'Meet link required'}), 400
    
    if meet_recorder.join_meet(meet_link):
        meet_recorder.start_recording()
        return jsonify({'status': 'Joined meeting and started recording'})
    return jsonify({'error': 'Failed to join meeting'}), 500

@app.route('/leave-meet', methods=['POST'])
def leave_meet():
    meet_recorder.stop_recording()
    return jsonify({'status': 'Left meeting and stopped recording'})

@app.route('/process', methods=['POST'])
def process_audio():
    global processing_progress
    processing_progress = 0
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        processing_progress = 10
        # Use Path for safer file path handling
        filename = Path(app.config['UPLOAD_FOLDER']) / file.filename
        filename = str(filename.resolve())
        file.save(filename)
        
        processing_progress = 30
        # Transcribe audio
        result = transcriber.transcribe(filename)
        transcript = result["text"]
        
        processing_progress = 50
        # Generate summary
        summary = summarizer(transcript, max_length=130, min_length=30)[0]['summary_text']
        
        processing_progress = 70
        # Enhanced analysis
        analysis = enhanced_processor.analyze_text(transcript)
        
        processing_progress = 90
        # Clean up
        os.remove(filename)
        
        processing_progress = 100
        return jsonify({
            'transcript': transcript,
            'summary': summary,
            'analysis': analysis
        })
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(filename):
            os.remove(filename)
        print(f"Error processing file: {str(e)}")  # Add logging
        return jsonify({'error': str(e)}), 500

class AudioDeviceManager:
    @staticmethod
    def check_audio_devices():
        try:
            devices = sd.query_devices()
            input_devices = [d for d in devices if d['max_input_channels'] > 0]
            if not input_devices:
                return False, "No audio input devices found"
            return True, "Audio devices available"
        except Exception as e:
            return False, f"Error checking audio devices: {str(e)}"

    @staticmethod
    def test_microphone(duration=3):
        """Test microphone and return audio level"""
        try:
            levels = []
            def callback(indata, frames, time, status):
                volume_norm = np.linalg.norm(indata) * 10
                levels.append(float(volume_norm))

            with sd.InputStream(callback=callback):
                sd.sleep(int(duration * 1000))
            
            avg_level = sum(levels) / len(levels) if levels else 0
            return {
                'working': avg_level > 1.0,
                'level': avg_level,
                'message': 'Audio detected' if avg_level > 1.0 else 'No audio detected'
            }
        except Exception as e:
            return {
                'working': False,
                'level': 0,
                'message': f'Error testing microphone: {str(e)}'
            }

class AutoRecorder:
    def __init__(self):
        self.is_recording = False
        self.current_thread = None
        self.frames = []
        self.samplerate = 44100
        self.channels = 2
        self.device_manager = AudioDeviceManager()

    def start(self):
        # Check devices before starting
        devices_ok, message = self.device_manager.check_audio_devices()
        if not devices_ok:
            raise RuntimeError(message)
            
        if not self.is_recording:
            self.is_recording = True
            self.frames = []
            self.current_thread = threading.Thread(target=self._record)
            self.current_thread.start()
            return True
        return False

    def stop(self):
        if self.is_recording:
            self.is_recording = False
            if self.current_thread:
                self.current_thread.join()
            return self._save_and_process()
        return False

    def _record(self):
        with sd.InputStream(samplerate=self.samplerate, channels=self.channels) as stream:
            while self.is_recording:
                data, overflowed = stream.read(1024)
                self.frames.append(data)

    def _save_and_process(self):
        if not self.frames:
            return False
        
        try:
            filename = f"recording_{int(time.time())}.wav"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            audio_data = np.concatenate(self.frames, axis=0)
            sf.write(filepath, audio_data, self.samplerate)
            
            # Process the recording with validation
            result = transcriber.transcribe(filepath)
            transcript = result["text"].strip()
            
            if not ProcessingManager.is_valid_transcript(transcript):
                return {
                    'transcript': transcript,
                    'summary': "Recording too short or no speech detected.",
                    'analysis': {
                        "sentiment": {"label": "NEUTRAL", "score": 0.5},
                        "action_items": [],
                        "speakers": [],
                        "meeting_type": {"type": "Unknown", "confidence": 0.0},
                        "key_points": []
                    }
                }
            
            # Dynamic max_length for summarization
            max_length = ProcessingManager.calculate_max_length(transcript)
            summary = summarizer(
                transcript, 
                max_length=max_length,
                min_length=min(10, max_length - 5)
            )[0]['summary_text']
            
            # Only perform analysis if transcript is long enough
            if len(transcript.split()) >= 10:
                analysis = enhanced_processor.analyze_text(transcript)
            else:
                analysis = enhanced_processor._get_default_analysis()
            
            # Clean up
            os.remove(filepath)
            
            return {
                'transcript': transcript,
                'summary': summary,
                'analysis': analysis
            }
        except Exception as e:
            print(f"Error processing recording: {str(e)}")
            return False

# Create auto recorder instance
auto_recorder = AutoRecorder()

# Remove the duplicate route and combine functionality
@app.route('/auto-record', methods=['POST'])
def auto_record():
    """Handle auto recording functionality"""
    try:
        data = request.json
        action = data.get('action')
        
        if action == 'start':
            # Check permissions first
            devices_ok, message = AudioDeviceManager.check_audio_devices()
            if not devices_ok:
                return jsonify({
                    'status': 'error',
                    'message': message
                }), 400

            if auto_recorder.start():
                return jsonify({
                    'status': 'success',
                    'message': 'Recording started'
                })
            return jsonify({
                'status': 'error',
                'message': 'Recording already in progress'
            }), 400
            
        elif action == 'stop':
            result = auto_recorder.stop()
            if result:
                return jsonify({
                    'status': 'success',
                    'message': 'Recording stopped and processed',
                    'data': result
                })
            return jsonify({
                'status': 'error',
                'message': 'No recording to stop or processing failed'
            }), 400
            
        return jsonify({
            'status': 'error',
            'message': 'Invalid action'
        }), 400
        
    except Exception as e:
        print(f"Auto-record error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# Add new route for checking permissions
@app.route('/check-permissions', methods=['GET'])
def check_permissions():
    """Check browser permissions and audio devices"""
    try:
        devices_ok, message = AudioDeviceManager.check_audio_devices()
        return jsonify({
            'status': 'success' if devices_ok else 'error',
            'devices_ok': devices_ok,
            'message': message
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'devices_ok': False,
            'message': str(e)
        }), 500

@app.route('/test-mic', methods=['POST'])
def test_mic():
    """Test microphone input"""
    try:
        result = AudioDeviceManager.test_microphone()
        return jsonify({
            'status': 'success',
            'working': result['working'],
            'level': result['level'],
            'message': result['message']
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    if not check_ffmpeg():
        print("Please ensure FFmpeg is installed in the correct location:")
        print(f"Expected path: {os.path.join(FFMPEG_PATH, 'ffmpeg.exe')}")
        exit(1)
    app.run(debug=True)
