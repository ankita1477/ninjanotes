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

# Add this near the top with other configurations
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

# Initialize models with API token
transcriber = whisper.load_model("base")
summarizer = pipeline("summarization", 
                     model="facebook/bart-large-cnn",
                     token=HUGGINGFACE_API_KEY)

sentiment_analyzer = pipeline("sentiment-analysis", 
                            model="distilbert-base-uncased-finetuned-sst-2-english",
                            token=HUGGINGFACE_API_KEY)

zero_shot_classifier = pipeline("zero-shot-classification",
                              token=HUGGINGFACE_API_KEY)

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
        nltk.download('punkt')
        self.action_keywords = ["todo", "action item", "follow up", "task", "deadline", "assign", "responsibility"]
        self.meeting_categories = [
            "Project Update", "Planning", "Review", "Brainstorming", 
            "Status Report", "Decision Making", "Training", "Client Meeting"
        ]

    def analyze_text(self, transcript):
        sentences = sent_tokenize(transcript)
        
        # Analyze sentiment
        sentiment = sentiment_analyzer(transcript[:512])[0]
        
        # Extract action items
        action_items = self._extract_action_items(sentences)
        
        # Identify speakers (basic approach)
        speakers = self._identify_speakers(transcript)
        
        # Classify meeting type
        meeting_type = self._classify_meeting_type(transcript)
        
        # Generate structured summary
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
        result = zero_shot_classifier(
            transcript[:512],
            candidate_labels=self.meeting_categories
        )
        return {
            "type": result["labels"][0],
            "confidence": float(result["scores"][0])
        }

    def _extract_key_points(self, sentences):
        return [s.strip() for s in sentences if len(s.split()) > 5][:5]  # Top 5 substantial sentences

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

if __name__ == '__main__':
    if not check_ffmpeg():
        print("Please ensure FFmpeg is installed in the correct location:")
        print(f"Expected path: {os.path.join(FFMPEG_PATH, 'ffmpeg.exe')}")
        exit(1)
    app.run(debug=True)
