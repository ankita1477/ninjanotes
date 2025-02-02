# API Documentation

## Endpoints

### 1. Process Audio
```http
POST /process
Content-Type: multipart/form-data

file: <audio-file>
```

### 2. Download & Process Video
```http
POST /download-video
Content-Type: application/json

{
    "video_url": "https://youtube.com/..."
}
```

### 3. Track Progress
```http
GET /progress
Accept: text/event-stream
```

### 4. Get Model Status
```http
GET /model-status
```

## Response Formats

### Success Response
```json
{
    "status": "success",
    "transcript": "...",
    "summary": "...",
    "segments": [...],
    "analysis": {
        "key_points": "...",
        "sentiment": {...},
        "metrics": {...}
    }
}
```

### Error Response
```json
{
    "status": "error",
    "message": "Error description"
}
```
