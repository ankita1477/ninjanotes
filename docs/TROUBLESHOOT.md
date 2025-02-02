# Troubleshooting Guide

## Common Issues

### 1. FFmpeg Not Found
```bash
# Windows
setx PATH "%PATH%;C:\ffmpeg-7.1-essentials_build\bin"
```

### 2. Port Already in Use
```python
# Change port in app.py
app.run(port=5001)
```

### 3. Memory Errors
- Close other applications
- Free up RAM
- Check file size (<16MB)
- Use smaller AI models

### 4. Model Loading Issues
```python
# Use smaller models
whisper.load_model("tiny")
```

### 5. Audio Processing Errors
- Check file format
- Ensure audio has content
- Verify FFmpeg installation
- Check file permissions
