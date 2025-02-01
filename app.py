from flask import Flask, render_template, request, jsonify, session
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
import whisper
from transformers import pipeline
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

# Update FFmpeg path to essentials build bin directory
FFMPEG_PATH = r"C:\ffmpeg-7.1-essentials_build\bin"
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
app.config['UPLOAD_FOLDER'] = r'C:\Users\KIIT0001\Documents\New folder (2)\ninjanotes\uploads'
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

# Initialize models
transcriber = whisper.load_model("base")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

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

meet_recorder = SimpleMeetRecorder()

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
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Use Path for safer file path handling
        filename = Path(app.config['UPLOAD_FOLDER']) / file.filename
        filename = str(filename.resolve())
        file.save(filename)
        
        # Transcribe audio
        result = transcriber.transcribe(filename)
        transcript = result["text"]
        
        # Generate summary
        summary = summarizer(transcript, max_length=130, min_length=30)[0]['summary_text']
        
        # Clean up
        os.remove(filename)
        
        return jsonify({
            'transcript': transcript,
            'summary': summary
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
