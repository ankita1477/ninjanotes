# 📥 Installation Guide

## System Requirements

- Windows 10/11 or Linux/MacOS
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space
- Python 3.8 or higher

## Step-by-Step Installation

### 1. Install Python
1. Download Python 3.8+ from [python.org](https://python.org)
2. During installation, check "Add Python to PATH"
3. Verify installation:
```bash
python --version
```

### 2. Install FFmpeg
#### Windows:
1. Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html)
2. Extract to `C:\ffmpeg-7.1-essentials_build`
3. Add to PATH:
   - Open System Properties > Advanced > Environment Variables
   - Add `C:\ffmpeg-7.1-essentials_build\bin` to Path

#### Linux:
```bash
sudo apt update
sudo apt install ffmpeg
```

#### MacOS:
```bash
brew install ffmpeg
```

### 3. Setup Project
```bash
# Clone repository
git clone https://github.com/ankita1477/ninjanotes.git
cd ninjanotes

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/MacOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install browser automation
playwright install

# Create required directory
mkdir uploads
```

### 4. Configuration
1. Copy `.env.example` to `.env`
2. Add your API keys:
```
OPENAI_API_KEY=your_key_here
HUGGINGFACE_API_KEY=your_key_here
```

### 5. Run Application
```bash
python app.py
```

Access the application at `http://localhost:5000`

## Verification Steps

1. Check FFmpeg installation:
```bash
ffmpeg -version
```

2. Verify Python packages:
```bash
pip list
```

3. Test browser automation:
```bash
playwright codegen google.com
```

## Troubleshooting

If you encounter any issues, check:
1. Python PATH configuration
2. FFmpeg PATH configuration
3. Virtual environment activation
4. Required ports availability (5000)
5. API keys configuration
