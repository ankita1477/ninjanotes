const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

export const uploadAudio = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_URL}/process`, {
    method: 'POST',
    body: formData,
  });
  return response.json();
};

export const downloadVideo = async (videoUrl) => {
  const response = await fetch(`${API_URL}/download-video`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ video_url: videoUrl }),
  });
  return response.json();
};

export const getSystemStatus = async () => {
  const response = await fetch(`${API_URL}/system-status`);
  return response.json();
};

export const getModelStatus = async () => {
  const response = await fetch(`${API_URL}/model-status`);
  return response.json();
};
