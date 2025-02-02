<div align="center">

# 🎯 NinjaNotes
> Your AI-Powered Audio Assistant

[![Windows Support](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)](INSTALLATION.md)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?logo=python&logoColor=white)](https://www.python.org)
[![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)

Transform audio files into text and summaries using AI 🎙️ ➡️ 📝

[Installation](INSTALLATION.md) • [Quick Start](#-quick-start) • [Features](#-features) • [Support](#-support)

</div>

---

## 🎯 Overview

NinjaNotes automatically processes audio files to generate:
- 📝 Accurate transcriptions
- ✨ Smart summaries
- 📊 Progress tracking
- 🔍 Detailed analysis

## ⚡ Quick Start

<table>
<tr>
<td width="50%">

### Windows Setup
```cmd
# 1. Install Requirements
- Install Python 3.8+
- Download FFmpeg
- Install Git

# 2. Clone & Setup
git clone https://github.com/ankita1477/ninjanotes.git
cd ninjanotes
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 3. Run
python app.py
```

</td>
<td width="50%">

### Requirements
- Windows 10/11
- Python 3.8+
- FFmpeg
- 4GB RAM
- Modern browser

[📚 Full Installation Guide](INSTALLATION.md)

</td>
</tr>
</table>

## 🎨 Features

<table>
<tr>
<td width="50%">

### Audio Processing
- **Supported Formats**
  - MP3
  - WAV
  - M4A
  - OGG
- **File Size**: Up to 16MB
- **Duration**: No limit
- **Quality**: Auto-validation

</td>
<td width="50%">

### AI Capabilities
- **Transcription**
  - 85-95% accuracy
  - Multiple languages
  - Noise handling
- **Summarization**
  - Key point extraction
  - Length control
  - Context awareness

</td>
</tr>
</table>

## 🔌 API Guide

```javascript
// 1. Upload Audio
POST /process
Content-Type: multipart/form-data
file: <audio-file>

// 2. Track Progress
GET /progress
Accept: text/event-stream

// 3. Get Status
GET /model-status
```

## 💻 Project Structure

```
ninjanotes/
├── app.py              # Main application
├── requirements.txt    # Dependencies
├── uploads/           # File storage
├── templates/         # Frontend
└── static/           # Assets
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Flask |
| Speech-to-Text | OpenAI Whisper |
| Summarization | Facebook BART |
| Audio Processing | FFmpeg |

## 🚨 Troubleshooting

<details>
<summary>Common Issues</summary>

1. **FFmpeg Not Found**
   ```cmd
   setx PATH "%PATH%;C:\ffmpeg-7.1-essentials_build\bin"
   ```

2. **Port 5000 Busy**
   ```python
   app.run(port=5001)
   ```

3. **Memory Error**
   - Close other applications
   - Free up RAM
   - Check file size (<16MB)

</details>

## 📚 Documentation

- [Installation Guide](INSTALLATION.md)
- [AI Models](AI_MODELS.md)
- [API Documentation](API.md)
- [Contributing Guide](CONTRIBUTING.md)

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Submit pull request

## 🆘 Support

Need help? [Open an Issue](https://github.com/ankita1477/ninjanotes/issues)

<div align="center">

---

Made with 🧡 by [ankita1477](https://github.com/ankita1477)

</div>
