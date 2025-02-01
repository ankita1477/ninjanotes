# 🎯 NinjaNotes - AI-Powered Meeting Assistant

Automatically join, record, transcribe, and summarize Google Meet sessions with AI intelligence.

## ✨ Key Features

- 🤖 AI-Powered Audio Processing
- 🎥 Automatic Google Meet Joining
- 🎙️ Smart Meeting Recording
- 📝 Real-time Transcription
- ✨ Intelligent Summarization
- 📊 Multiple Format Support (MP3, WAV, M4A, OGG)

## 🧠 AI Models

### OpenAI Whisper (Speech-to-Text)
- Model Size: Base (1GB)
- Accuracy: 85-95%
- Speed: Real-time processing
- Features:
  - Multilingual support
  - Punctuation & formatting
  - Background noise handling

### Facebook BART (Summarization)
- Model: facebook/bart-large-cnn
- Accuracy: 90%+
- Processing: <2s per text
- Features:
  - Smart meeting summarization
  - Key point extraction
  - Length control (30-130 words)

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8+
- FFmpeg
- Chrome/Chromium browser

### 2. Clone the repository
```bash
git clone https://github.com/yourusername/ninjanotes.git
cd ninjanotes
```

### 3. Install FFmpeg
- Download FFmpeg essentials build
- Extract to C:\ffmpeg-7.1-essentials_build
- Add bin directory to PATH

### 4. Set up Python environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
playwright install
```

### 5. Create required directories
```bash
mkdir uploads
```

### 6. Run the application
```bash
python app.py
```

7. Open `http://localhost:5000` in your browser

## Usage

### Recording Google Meet Sessions

1. Start the application
2. Enter Google Meet link in the interface
3. Click "Join & Record"
4. Wait for automatic joining and recording
5. Click "Leave Meeting" when done

### Processing Audio Files

1. Upload supported audio file
2. Click "Process Audio"
3. View transcript and summary

## Project Structure

```
ninjanotes/
├── app.py             # Main Flask application
├── requirements.txt   # Python dependencies
├── uploads/          # Temporary audio storage
├── templates/        # HTML templates
│   └── index.html    # Main interface
└── README.md         # This file
```

## Technologies Used

- **Flask**: Web framework
- **Playwright**: Browser automation
- **Whisper**: Speech-to-text
- **BART**: Text summarization
- **Bootstrap**: UI styling

## Troubleshooting

### Common Issues

1. **FFmpeg not found**
   - Ensure FFmpeg is installed in the correct location
   - Check system PATH includes FFmpeg bin directory

2. **Audio recording issues**
   - Check microphone permissions
   - Verify sound input device is working

3. **Meet joining fails**
   - Check internet connection
   - Verify Meet link format
   - Ensure browser permissions are granted

## Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## License

MIT License - feel free to use for any purpose.

## Acknowledgments

- OpenAI Whisper for transcription
- Facebook BART for summarization
- Microsoft Playwright for automation
