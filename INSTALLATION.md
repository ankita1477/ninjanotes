<div align="center">

# 📥 NinjaNotes Windows Installation Guide
> Easy setup guide for Windows users

![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)

</div>

## 📋 Windows System Requirements

- Windows 10 or 11
- 4GB RAM minimum
- Python 3.8 or higher
- Internet connection
- Chrome/Edge/Firefox browser

## 🚀 Step-by-Step Windows Installation

### Step 1: Install Python
1. Download Python 3.8+ from [python.org](https://python.org)
2. Run the installer:
   - ✅ Check "Add Python to PATH"
   - ✅ Check "Install for all users"
   ![Python Install](docs/python-install.png)
3. Verify in Command Prompt:
   ```cmd
   python --version
   pip --version
   ```

### Step 2: Install FFmpeg
1. Download [FFmpeg Essentials Build](https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.7z)
2. Extract the archive:
   - Right-click → Extract All
   - Extract to: `C:\ffmpeg-7.1-essentials_build`
3. Add to System PATH:
   - Press `Win + X` → System
   - Click "Advanced system settings"
   - Click "Environment Variables"
   - Under "System variables", select "Path"
   - Click "Edit" → "New"
   - Add: `C:\ffmpeg-7.1-essentials_build\bin`
   - Click "OK" on all windows
4. Verify in Command Prompt:
   ```cmd
   ffmpeg -version
   ```

### Step 3: Download NinjaNotes
```cmd
:: Clone repository
git clone https://github.com/ankita1477/ninjanotes.git
cd ninjanotes
```

### Step 4: Setup Python Environment
```cmd
:: Create virtual environment
python -m venv venv

:: Activate environment
venv\Scripts\activate

:: Install dependencies
pip install -r requirements.txt
```

### Step 5: Create Storage Directory
```cmd
:: Create uploads folder
mkdir uploads
```

### Step 6: Run Application
```cmd
:: Start NinjaNotes
python app.py
```

### Step 7: Access Application
1. Open your browser
2. Visit: http://localhost:5000
3. Upload an audio file to test

## ❌ Common Windows Issues

### Python Not Found
```cmd
:: Add Python to PATH manually
setx PATH "%PATH%;C:\Users\YourUsername\AppData\Local\Programs\Python\Python38"
```

### FFmpeg Not Found
```cmd
:: Verify FFmpeg path
echo %PATH%
:: Should include: C:\ffmpeg-7.1-essentials_build\bin
```

### Port 5000 Busy
```python
# In app.py, modify:
app.run(port=5001)  # Change port number
```

### Installation Checklist
- [ ] Python 3.8+ installed
- [ ] FFmpeg installed and in PATH
- [ ] Virtual environment active
- [ ] Dependencies installed
- [ ] Application running

## 🆘 Need Help?

1. Check Windows error logs
2. Verify Python PATH: `echo %PATH%`
3. Check FFmpeg installation
4. [Open an Issue](https://github.com/ankita1477/ninjanotes/issues)

## 🔄 Starting the App
```cmd
:: Quick start commands
cd ninjanotes
venv\Scripts\activate
python app.py
```

Need more help? Contact [Support](https://github.com/ankita1477)
