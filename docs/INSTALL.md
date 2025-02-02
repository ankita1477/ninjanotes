# Installation Guide

## Prerequisites
- Python 3.8+
- FFmpeg
- Git
- 4GB RAM minimum

## Step-by-Step Setup

### 1. Install Dependencies
```bash
# Windows
# Install FFmpeg
# Download from: https://ffmpeg.org/download.html
setx PATH "%PATH%;C:\ffmpeg-7.1-essentials_build\bin"

# Install Python
# Download from: https://www.python.org/downloads/
```

### 2. Clone & Setup
```bash
# Clone repository
git clone https://github.com/ankita1477/ninjanotes.git
cd ninjanotes

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Configuration
```bash
# Create .env file
cp .env.example .env

# Add your Hugging Face API key
# Edit .env file:
HUGGINGFACE_API_KEY=your_key_here
```

### 4. Run Application
```bash
python app.py
```

## Troubleshooting
- [Common Issues](TROUBLESHOOT.md)
- [FAQ](FAQ.md)
