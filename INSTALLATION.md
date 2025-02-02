<div align="center">

# 📥 NinjaNotes Installation Guide

> Step-by-step setup instructions

</div>

## 📋 Prerequisites

First, ensure you have:
- [ ] Python 3.8 or higher
- [ ] Git
- [ ] FFmpeg
- [ ] 4GB RAM minimum
- [ ] Modern web browser

## 🚀 Installation Steps

### Step 1: Install Python
1. Download Python 3.8+ from [python.org](https://python.org)
2. Run installer
3. ✅ Check "Add Python to PATH"
4. Verify installation:
   ```bash
   python --version
   pip --version
   ```

### Step 2: Install FFmpeg

#### Windows:
1. Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html)
2. Extract to `C:\ffmpeg-7.1-essentials_build`
3. Add to PATH:
   - Open System Properties
   - Click "Environment Variables"
   - Edit "Path"
   - Add `C:\ffmpeg-7.1-essentials_build\bin`
4. Verify:
   ```bash
   ffmpeg -version
   ```

#### Linux:
```bash
# Update package list
sudo apt update

# Install FFmpeg
sudo apt install ffmpeg -y

# Verify installation
ffmpeg -version
```

#### MacOS:
```bash
# Install Homebrew if needed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install FFmpeg
brew install ffmpeg

# Verify installation
ffmpeg -version
```

### Step 3: Clone Repository
```bash
# Clone the repository
git clone https://github.com/ankita1477/ninjanotes.git

# Navigate to project directory
cd ninjanotes
```

### Step 4: Set Up Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/MacOS:
source venv/bin/activate
```

### Step 5: Install Dependencies
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

### Step 6: Create Required Directories
```bash
# Create uploads directory
mkdir uploads
```

### Step 7: Run Application
```bash
# Start the application
python app.py
```

### Step 8: Verify Installation
1. Open browser
2. Visit: http://localhost:5000
3. Upload test audio file
4. Check transcription

## 🔧 Troubleshooting

### Common Issues

1. **Python Command Not Found**
   ```bash
   # Windows: Add to PATH manually
   # Linux/MacOS: Use python3 instead
   python3 --version
   ```

2. **FFmpeg Not Found**
   ```bash
   # Check PATH
   echo $PATH
   # or on Windows
   echo %PATH%
   ```

3. **Port 5000 Already in Use**
   ```python
   # In app.py, modify:
   app.run(port=5001)
   ```

4. **Virtual Environment Issues**
   ```bash
   # If venv fails, try:
   python -m pip install --upgrade virtualenv
   python -m virtualenv venv
   ```

## ✅ Success Checklist

- [ ] Python installed & in PATH
- [ ] FFmpeg installed & in PATH
- [ ] Repository cloned
- [ ] Virtual environment active
- [ ] Dependencies installed
- [ ] Application running
- [ ] Web interface accessible

Need help? [Open an Issue](https://github.com/ankita1477/ninjanotes/issues)
