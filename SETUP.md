# 🚀 Quick Setup Guide for NinjaNotes

## Windows Setup (5 minutes)

### 1️⃣ Install Required Software
```powershell
# Download and install Python 3.8+ from python.org
# Download FFmpeg from ffmpeg.org
# Extract FFmpeg to C:\ffmpeg-7.1-essentials_build
```

### 2️⃣ One-Command Setup
Copy and paste this command in PowerShell:
```powershell
git clone https://github.com/ankita1477/ninjanotes.git; cd ninjanotes; python -m venv venv; .\venv\Scripts\activate; pip install -r requirements.txt; mkdir uploads; python app.py
```

## Linux/MacOS Setup (3 minutes)

### 1️⃣ Install Dependencies
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install -y python3.8 python3-pip ffmpeg git

# MacOS
brew install python ffmpeg git
```

### 2️⃣ One-Command Setup
```bash
git clone https://github.com/ankita1477/ninjanotes.git && cd ninjanotes && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt && mkdir uploads && python app.py
```

## ✅ Verify Installation

1. Open browser: http://localhost:5000
2. Upload test audio file (MP3/WAV)
3. Check transcription result

## 🔧 Troubleshooting

### Common Issues

1. **FFmpeg Error**
   ```bash
   # Windows: Add to PATH
   setx PATH "%PATH%;C:\ffmpeg-7.1-essentials_build\bin"
   
   # Linux/MacOS: Verify installation
   ffmpeg -version
   ```

2. **Port 5000 Busy**
   ```python
   # In app.py, change:
   app.run(port=5001)  # Use different port
   ```

3. **Python venv Issues**
   ```bash
   # Windows
   python -m pip install --upgrade pip
   python -m pip install virtualenv
   
   # Linux/MacOS
   python3 -m pip install --upgrade pip
   python3 -m pip install virtualenv
   ```

## 📝 Requirements.txt
```
flask==2.0.1
openai-whisper==20230314
transformers==4.28.1
torch==2.0.0
soundfile==0.12.1
pydub==0.25.1
