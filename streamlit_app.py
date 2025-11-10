import os
# Fix for Streamlit deployment
os.environ['MPLCONFIGDIR'] = '/tmp/.matplotlib'

import streamlit as st
import numpy as np
import torch
# ... rest of your code
import streamlit as st
import numpy as np
import torch
import librosa
import matplotlib.pyplot as plt
import tempfile
import os
from joblib import load
import json
import torch.nn as nn
import torch.nn.functional as F

# Your existing model classes (copy from app.py)
class BDNN_Head(nn.Module):
    def __init__(self, in_dim=1521, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x).squeeze(-1)

class CNN_VAD(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.25)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 39, 39)
            x = F.relu(self.conv1(dummy)); x = self.pool(x)
            x = F.relu(self.conv2(x)); x = self.pool(x); x = self.dropout1(x)
            conv_out = x.view(1, -1).shape[1]
        self.fc1 = nn.Linear(conv_out, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = F.relu(self.conv1(x)); x = self.pool(x)
        x = F.relu(self.conv2(x)); x = self.pool(x); x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x)); x = self.dropout2(x); x = self.fc2(x)
        return self.sigmoid(x).squeeze(-1)

class MetaMLP(nn.Module):
    def __init__(self, in_dim=2, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x).squeeze(-1)

class HybridVADAnalyzer:
    def __init__(self, model_dir="hybrid_model"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir
        self.load_models()
    
    def load_models(self):
        with open(f"{self.model_dir}/hybrid_config.json", "r") as f:
            config = json.load(f)
        self.win_ctx = config["win_ctx"]
        self.n_feats = config["n_feats"]
        self.win_total = 2 * self.win_ctx + 1
        self.flat_dim = self.win_total * self.n_feats
        
        self.scaler = load(f"{self.model_dir}/scaler_ctx.joblib")
        self.bdnn = BDNN_Head(in_dim=self.flat_dim)
        self.cnn = CNN_VAD()
        self.meta = MetaMLP()
        
        self.bdnn.load_state_dict(torch.load(f"{self.model_dir}/bdnn.pth", map_location=self.device))
        self.cnn.load_state_dict(torch.load(f"{self.model_dir}/cnn.pth", map_location=self.device))
        self.meta.load_state_dict(torch.load(f"{self.model_dir}/meta.pth", map_location=self.device))
        
        self.bdnn.to(self.device).eval()
        self.cnn.to(self.device).eval()
        self.meta.to(self.device).eval()

    def extract_features(self, audio_path):
        y, sr = librosa.load(audio_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=160)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        features = np.vstack([mfcc, delta, delta2]).T
        return features, sr, y

    def add_context(self, features):
        padded = np.pad(features, ((self.win_ctx, self.win_ctx), (0, 0)), mode='edge')
        context_features = []
        for i in range(self.win_ctx, len(features) + self.win_ctx):
            window = padded[i-self.win_ctx:i+self.win_ctx+1].flatten()
            context_features.append(window)
        return np.array(context_features)

    def predict_vad(self, audio_path, threshold=0.5, min_duration=0.1):
        features, sr, audio_data = self.extract_features(audio_path)
        features_ctx = self.add_context(features)
        features_scaled = self.scaler.transform(features_ctx)
        
        features_flat = torch.tensor(features_scaled, dtype=torch.float32).to(self.device)
        features_cnn = features_flat.reshape(-1, 1, self.win_total, self.n_feats)
        
        with torch.no_grad():
            bdnn_probs = self.bdnn(features_flat).cpu().numpy()
            cnn_probs = self.cnn(features_cnn).cpu().numpy()
            meta_input = torch.tensor(np.column_stack([bdnn_probs, cnn_probs]), dtype=torch.float32).to(self.device)
            final_probs = self.meta(meta_input).cpu().numpy()
        
        times = librosa.frames_to_time(range(len(final_probs)), sr=sr, hop_length=160)
        segments = []
        if len(final_probs) > 0:
            current_label = "silence" if final_probs[0] < threshold else "speech"
            start_time = float(times[0])
            
            for i in range(1, len(final_probs)):
                new_label = "silence" if final_probs[i] < threshold else "speech"
                if new_label != current_label:
                    duration = times[i] - start_time
                    if duration >= min_duration:
                        start_frame = int(start_time * 100)
                        end_frame = int(times[i] * 100)
                        segment_probs = final_probs[start_frame:end_frame]
                        confidence = float(np.mean(segment_probs)) if len(segment_probs) > 0 else 0.0
                        
                        segments.append({
                            'start': float(start_time), 'end': float(times[i]),
                            'label': current_label, 'confidence': confidence
                        })
                        start_time = times[i]
                        current_label = new_label
            
            # Final segment
            duration = times[-1] - start_time
            if duration >= min_duration:
                start_frame = int(start_time * 100)
                end_frame = int(times[-1] * 100)
                segment_probs = final_probs[start_frame:end_frame]
                confidence = float(np.mean(segment_probs)) if len(segment_probs) > 0 else 0.0
                
                segments.append({
                    'start': float(start_time), 'end': float(times[-1]),
                    'label': current_label, 'confidence': confidence
                })
        
        speech_time = sum(s['end']-s['start'] for s in segments if s['label'] == 'speech')
        total_time = segments[-1]['end'] if segments else 0
        speech_ratio = speech_time / total_time if total_time > 0 else 0
        
        return {
            'segments': segments,
            'speech_ratio': speech_ratio,
            'total_duration': total_time,
            'speech_duration': speech_time,
            'silence_duration': total_time - speech_time
        }

# Streamlit UI
st.set_page_config(page_title="Hybrid VAD Analyzer", page_icon="🎵", layout="wide")

st.title("🎵 Hybrid VAD Analyzer")
st.markdown("Upload an audio file to detect speech and silence segments using advanced AI")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Detection Threshold", 0.1, 0.9, 0.5, 0.05)
    min_duration = st.slider("Minimum Segment Duration (s)", 0.05, 0.5, 0.1, 0.05)

# File upload
uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'm4a', 'flac'])

if uploaded_file is not None:
    with st.spinner("Analyzing audio..."):
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Initialize analyzer and predict
            analyzer = HybridVADAnalyzer()
            results = analyzer.predict_vad(tmp_path, threshold, min_duration)
            
            # Display results
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Speech Ratio", f"{results['speech_ratio']:.1%}")
            with col2:
                st.metric("Total Duration", f"{results['total_duration']:.2f}s")
            with col3:
                st.metric("Speech Duration", f"{results['speech_duration']:.2f}s")
            with col4:
                st.metric("Silence Duration", f"{results['silence_duration']:.2f}s")
            
            # Display segments
            st.subheader("Segments")
            for i, seg in enumerate(results['segments']):
                color = "🟢" if seg['label'] == 'speech' else "🔴"
                st.write(f"{color} **{seg['label'].upper()}** | "
                        f"Time: {seg['start']:.2f}s - {seg['end']:.2f}s | "
                        f"Duration: {seg['end']-seg['start']:.2f}s | "
                        f"Confidence: {seg['confidence']:.1%}")
            
            # Visualization
            st.subheader("Waveform Visualization")
            y, sr = librosa.load(tmp_path)
            fig, ax = plt.subplots(figsize=(12, 3))
            librosa.display.waveshow(y, sr=sr, alpha=0.7, color='blue', ax=ax)
            
            for seg in results['segments']:
                color = 'green' if seg['label'] == 'speech' else 'red'
                ax.axvspan(seg['start'], seg['end'], color=color, alpha=0.3)
            
            ax.set_title("Voice Activity Detection Results")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
        finally:
            # Clean up
            os.unlink(tmp_path)

st.markdown("---")
st.markdown("Powered by Hybrid VAD Model | Built with Streamlit")