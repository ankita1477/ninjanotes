# 🎯 NinjaNotes
> AI-Powered Audio Processing & Summarization Tool

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![FFmpeg](https://img.shields.io/badge/FFmpeg-Required-orange)

## 📚 Table of Contents
1. [Features](#-features)
2. [Quick Start](#-quick-start)
3. [Installation](#-installation)
4. [Usage Guide](#-usage-guide)
5. [Technical Details](#-technical-details)
6. [Troubleshooting](#-troubleshooting)
7. [Contributing](#-contributing)

## ✨ Features
### Core Capabilities
- 🎙️ Multi-format Audio Processing (MP3, WAV, M4A, OGG)
- 📝 Advanced Speech-to-Text Transcription
- ✨ Intelligent Text Summarization
- 📊 Real-time Progress Tracking

### AI Models
| Model | Task | Accuracy | Speed |
|-------|------|----------|-------|
| OpenAI Whisper | Transcription | 85-95% | Real-time |
| Facebook BART | Summarization | 90%+ | <2s/text |

## 🚀 Quick Start
```bash
git clone https://github.com/ankita1477/ninjanotes.git
cd ninjanotes
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## 📥 Installation
Detailed installation steps in [INSTALLATION.md](INSTALLATION.md)

### System Requirements
- Python 3.8+
- FFmpeg
- 4GB RAM
- 2GB disk space
- Modern web browser

## 📝 Usage Guide

### Processing Audio Files
1. Launch application
2. Upload audio file (MP3/WAV/M4A/OGG)
3. Monitor progress indicators
4. View transcript and summary

### API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main interface |
| `/process` | POST | Process audio |
| `/progress` | GET | Track progress |
| `/model-status` | GET | Model status |

## 🔧 Technical Details

### Project Structure
```
ninjanotes/
├── app.py              # Main application
├── requirements.txt    # Dependencies
├── uploads/           # Audio storage
├── templates/         # Frontend
└── README.md         # Documentation
```

### Technologies
- Backend: Flask
- AI: Whisper, BART
- Audio: FFmpeg
- Frontend: Bootstrap

## ⚠️ Troubleshooting

### Common Issues
| Issue | Solution |
|-------|----------|
| FFmpeg missing | [Install Guide](INSTALLATION.md#2-install-ffmpeg) |
| Port in use | Change port in app.py |
| Memory error | Free up RAM |
| Upload fails | Check file size (<16MB) |

### Error Handling
- Automatic format validation
- Audio quality checks
- Model error recovery
- Progress monitoring

## 🤝 Contributing
1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Open pull request

## 📄 License
MIT License

## 🙏 Acknowledgments
- OpenAI Whisper Team
- Facebook BART Team
- FFmpeg Project
