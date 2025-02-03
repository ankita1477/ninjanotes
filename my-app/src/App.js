import React, { useState, useEffect } from 'react';
import { uploadAudio, downloadVideo, getSystemStatus } from './services/api';
import './App.css';

const UploadSection = ({ onFileChange, onVideoUrlChange, onAudioUpload, onVideoDownload, loading, videoUrl, selectedFile }) => (
  <div className="upload-container">
    <div className="upload-section">
      <div className="upload-card">
        <h2>Audio Upload</h2>
        <div className="upload-area">
          <label className="file-input">
            <input type="file" accept="audio/*" onChange={onFileChange} />
            <span>{selectedFile ? selectedFile.name : 'Choose Audio File'}</span>
          </label>
          {selectedFile && (
            <div className="file-info">
              <span className="file-name">
                📎 {selectedFile.name}
              </span>
              <span className="file-size">
                ({(selectedFile.size / (1024 * 1024)).toFixed(2)} MB)
              </span>
            </div>
          )}
          <button 
            className="primary-button" 
            onClick={onAudioUpload} 
            disabled={loading || !selectedFile}
          >
            {loading ? (
              <><span className="spinner"></span>Processing...</>
            ) : (
              'Process Audio'
            )}
          </button>
        </div>
      </div>
    </div>

    <div className="upload-section">
      <div className="upload-card">
        <h2>Video URL</h2>
        <div className="upload-area">
          <input
            type="text"
            value={videoUrl}
            onChange={onVideoUrlChange}
            placeholder="Paste YouTube URL"
            className="url-input"
          />
          <button className="primary-button" onClick={onVideoDownload} disabled={loading}>
            {loading ? 'Processing...' : 'Process Video'}
          </button>
        </div>
      </div>
    </div>
  </div>
);

const ChaptersList = ({ chapters }) => (
  <div className="chapters">
    <h4>Chapters:</h4>
    <div className="chapters-list">
      {chapters.map((chapter, index) => (
        <div key={index} className="chapter-item">
          <span className="timestamp">[{chapter.start} - {chapter.end}]</span>
          <span className="text">{chapter.text}</span>
        </div>
      ))}
    </div>
  </div>
);

const Stats = ({ stats, duration }) => (
  <div className="stats-card">
    <h4>Statistics</h4>
    <div className="stats-grid">
      <div className="stat-item">
        <div className="stat-icon">⏱️</div>
        <div className="stat-info">
          <span className="stat-label">Duration</span>
          <span className="stat-value">{duration}</span>
        </div>
      </div>
      <div className="stat-item">
        <div className="stat-icon">📝</div>
        <div className="stat-info">
          <span className="stat-label">Words</span>
          <span className="stat-value">{stats.wordCount}</span>
        </div>
      </div>
      <div className="stat-item">
        <div className="stat-icon">👥</div>
        <div className="stat-info">
          <span className="stat-label">Speakers</span>
          <span className="stat-value">{stats.speakerCount}</span>
        </div>
      </div>
      <div className="stat-item">
        <div className="stat-icon">💭</div>
        <div className="stat-info">
          <span className="stat-label">Sentiment</span>
          <span className={`stat-value sentiment-${stats.sentiment}`}>
            {stats.sentiment}
          </span>
        </div>
      </div>
    </div>
  </div>
);

const Analysis = ({ analysis }) => (
  <div className="analysis">
    <h4>Analysis:</h4>
    {analysis.topics.length > 0 && (
      <div className="topics">
        <h5>Key Topics:</h5>
        <div className="topics-list">
          {analysis.topics.map((topic, index) => (
            <div key={index} className="topic-item">
              <span className="topic-text">{topic.text}</span>
              <span className="topic-score">{topic.score}%</span>
            </div>
          ))}
        </div>
      </div>
    )}
  </div>
);

function App() {
  const [file, setFile] = useState(null);
  const [videoUrl, setVideoUrl] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [systemStatus, setSystemStatus] = useState(null);

  useEffect(() => {
    const pollSystemStatus = async () => {
      try {
        const status = await getSystemStatus();
        setSystemStatus(status);
      } catch (err) {
        console.error('Error fetching system status:', err);
      }
    };

    const interval = setInterval(pollSystemStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
    setError(null);
  };

  const handleVideoUrlChange = (event) => {
    setVideoUrl(event.target.value);
    setError(null);
  };

  const handleAudioUpload = async () => {
    if (!file) {
      setError('Please select a file');
      return;
    }

    setLoading(true);
    try {
      const data = await uploadAudio(file);
      setResult(data);
    } catch (err) {
      setError('Error processing audio file');
    }
    setLoading(false);
  };

  const handleVideoDownload = async () => {
    if (!videoUrl) {
      setError('Please enter a video URL');
      return;
    }

    setLoading(true);
    try {
      const data = await downloadVideo(videoUrl);
      setResult(data);
    } catch (err) {
      setError('Error processing video');
    }
    setLoading(false);
  };

  const renderResults = (result) => {
    if (!result) return null;
    
    return (
      <div className="results-container">
        {result.stats && result.duration && (
          <Stats stats={result.stats} duration={result.duration} />
        )}

        <div className="content-card">
          <div className="card-header">
            <h4>Summary</h4>
          </div>
          <p className="summary-text">{result.summary}</p>
        </div>

        {result.chapters && result.chapters.length > 0 && (
          <div className="content-card">
            <div className="card-header">
              <h4>Chapters</h4>
            </div>
            <ChaptersList chapters={result.chapters} />
          </div>
        )}

        <div className="content-card">
          <div className="card-header">
            <h4>Full Transcript</h4>
          </div>
          <p className="transcript-text">{result.transcript}</p>
        </div>

        {result.analysis && (
          <div className="content-card">
            <div className="card-header">
              <h4>Content Analysis</h4>
            </div>
            <Analysis analysis={result.analysis} />
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="App">
      <nav className="app-nav">
        <div className="nav-content">
          <h1>NinjaNotes</h1>
          <span className="subtitle">AI-Powered Transcription</span>
        </div>
      </nav>

      <main className="app-main">
        <UploadSection
          onFileChange={handleFileChange}
          onVideoUrlChange={handleVideoUrlChange}
          onAudioUpload={handleAudioUpload}
          onVideoDownload={handleVideoDownload}
          loading={loading}
          videoUrl={videoUrl}
          selectedFile={file}
        />

        {loading && (
          <div className="loader-container">
            <div className="loader-spinner"></div>
            <p>Processing your content...</p>
          </div>
        )}
        
        {error && <div className="error-message">{error}</div>}
        {result && renderResults(result)}

        {systemStatus && (
          <div className="system-status">
            <div className="status-item">
              <span className="status-label">System Load:</span>
              <span className="status-value">{systemStatus.system.cpu.percent}%</span>
            </div>
            <div className="status-item">
              <span className="status-label">Memory:</span>
              <span className="status-value">{systemStatus.system.memory.percent}%</span>
            </div>
            <div className="status-item">
              <span className="status-label">Model:</span>
              <span className="status-value">{systemStatus.model.status}</span>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
