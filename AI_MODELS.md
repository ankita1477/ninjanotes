# 🤖 AI Models Used in NinjaNotes

## Speech-to-Text: OpenAI Whisper
![Whisper Model](https://img.shields.io/badge/Whisper-Base-blue)

**Purpose**: Transcription of audio to text
- Model Size: Base (1GB)
- Languages: Multilingual support
- Features:
  - ✨ High accuracy transcription
  - 🌍 Multiple language detection
  - 🎯 Punctuation and formatting
  - ⚡ Offline processing

## Text Summarization: Facebook BART
![BART Model](https://img.shields.io/badge/BART-Large_CNN-green)

**Purpose**: Generate concise summaries
- Model: facebook/bart-large-cnn
- Features:
  - 📝 Abstractive summarization
  - 🎯 Meeting-focused summaries
  - 📊 Length control (30-130 words)
  - 🔍 Key point extraction

## Performance Metrics

| Model | Task | Speed | Accuracy | Memory Usage |
|-------|------|-------|----------|--------------|
| Whisper Base | Transcription | ~1x RT* | 85-95% | 1GB |
| BART Large CNN | Summarization | <2s/text | 90%+ | 1.5GB |

*RT = Real-time (1 minute audio ≈ 1 minute processing)

## Model Configuration

### Whisper Settings
