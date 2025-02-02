<div align="center">

# 📥 NinjaNotes Installation Guide

> Get up and running in minutes!

</div>

## 📋 Prerequisites Checklist

<table>
<tr>
<td>

- [ ] Python 3.8+
- [ ] FFmpeg
- [ ] 4GB RAM
- [ ] Git
- [ ] Modern Browser

</td>
<td>

Check with these commands:
```bash
python --version
ffmpeg -version
git --version
```

</td>
</tr>
</table>

## 🚀 Quick Install

### Windows One-Line Install
```powershell
Set-ExecutionPolicy RemoteSignedUser; iwr -useb https://raw.githubusercontent.com/ankita1477/ninjanotes/main/install.ps1 | iex
```

### Linux/MacOS One-Line Install
```bash
curl -fsSL https://raw.githubusercontent.com/ankita1477/ninjanotes/main/install.sh | bash
```

## 📖 Step-by-Step Guide

<details>
<summary>1️⃣ Python Installation</summary>

Download Python 3.8+ from [python.org](https://python.org)
```bash
# Verify installation
python --version
pip --version
```
</details>

<details>
<summary>2️⃣ FFmpeg Setup</summary>

#### 🪟 Windows
1. Download from [ffmpeg.org](https://ffmpeg.org/download.html)
2. Extract to `C:\ffmpeg-7.1-essentials_build`
3. Add to PATH

#### 🐧 Linux
```bash
sudo apt update && sudo apt install ffmpeg -y
```

#### 🍎 MacOS
```bash
brew install ffmpeg
```
</details>

<details>
<summary>3️⃣ Project Setup</summary>

```bash
# Clone & Setup
git clone https://github.com/ankita1477/ninjanotes.git
cd ninjanotes

# Virtual Environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Dependencies
pip install -r requirements.txt

# Create Directories
mkdir uploads
```
</details>

## ✅ Verification

Run these tests to verify your installation:

```bash
# 1. Test Python
python -c "print('Python works!')"

# 2. Test FFmpeg
ffmpeg -version

# 3. Test Application
python app.py
```

## 🔧 Troubleshooting

<details>
<summary>Common Issues</summary>

| Error | Solution |
|-------|----------|
| `Python not found` | Add Python to PATH |
| `FFmpeg missing` | Check FFmpeg installation |
| `Port 5000 in use` | Change port in app.py |

</details>

<div align="center">

---

Need help? [Open an Issue](https://github.com/ankita1477/ninjanotes/issues)

</div>
