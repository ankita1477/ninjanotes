from flask import Flask, render_template, request, jsonify
import whisper
from transformers import pipeline
import os
import subprocess
from pathlib import Path

# Update FFmpeg path to point directly to the bin directory
FFMPEG_PATH = r"C:\Users\KIIT0001\Downloads\ffmpeg-7.1 (1)\ffmpeg-7.1\bin"
os.environ["PATH"] = FFMPEG_PATH + os.pathsep + os.environ["PATH"]

def check_ffmpeg():
    try:
        # Check for ffmpeg.exe directly in the bin directory
        ffmpeg_exe = os.path.join(FFMPEG_PATH, 'ffmpeg.exe')
        if not os.path.exists(ffmpeg_exe):
            print(f"FFmpeg executable not found at: {ffmpeg_exe}")
            return False
        subprocess.run([ffmpeg_exe, '-version'], capture_output=True, check=True)
        print(f"FFmpeg found successfully at: {ffmpeg_exe}")
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

@app.route('/')
def index():
    return render_template('index.html')

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
