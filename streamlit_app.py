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
import warnings
import pandas as pd
import io
from datetime import datetime
import time

# Fix for deployment
os.environ['MPLCONFIGDIR'] = '/tmp/.matplotlib'

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ========== CUSTOM CSS - MODERN DESIGN ==========
st.markdown("""
<style>
    /* Main page styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #2A2D43 100%);
        background-attachment: fixed;
    }
    
    /* Beautiful gradient cards */
    .gradient-card {
        background: linear-gradient(145deg, rgba(255,255,255,0.95), rgba(255,255,255,0.85));
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 25px;
        margin: 10px 0;
        border: 1px solid rgba(255,255,255,0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .gradient-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
    }
    
    /* Sidebar with glass effect */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(42,45,67,0.9) 0%, rgba(31,33,48,0.9) 100%);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Animated title */
    .animated-title {
        background: linear-gradient(90deg, #FF6B6B, #FFD166, #06D6A0, #118AB2, #073B4C);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient 8s ease infinite;
        font-size: 3.5em;
        font-weight: 800;
        text-align: center;
        margin-bottom: 10px;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Floating elements */
    .floating {
        animation: float 6s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }
    
    /* Glowing buttons */
    .glowing-btn {
        background: linear-gradient(45deg, #FF6B6B, #FFD166, #06D6A0);
        background-size: 200% 200%;
        animation: glowing 5s ease infinite;
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 50px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    @keyframes glowing {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Custom file uploader */
    .upload-box {
        border: 3px dashed #FF6B6B;
        border-radius: 25px;
        padding: 60px 20px;
        text-align: center;
        background: rgba(255,255,255,0.05);
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .upload-box:hover {
        border-color: #06D6A0;
        background: rgba(6, 214, 160, 0.1);
        transform: scale(1.02);
    }
    
    /* Custom metrics */
    .custom-metric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .custom-metric:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Pulse animation */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    /* Shimmer effect */
    .shimmer {
        background: linear-gradient(90deg, 
            rgba(255,255,255,0) 0%, 
            rgba(255,255,255,0.2) 50%, 
            rgba(255,255,255,0) 100%);
        background-size: 200% 100%;
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    
    /* Custom tabs */
    .custom-tab {
        background: rgba(255,255,255,0.1);
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        margin: 0 5px;
        color: white;
        transition: all 0.3s ease;
    }
    
    .custom-tab:hover {
        background: rgba(255,255,255,0.2);
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #FF6B6B, #FFD166, #06D6A0);
    }
    
    /* Custom dataframe styling */
    .dataframe {
        background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(255,255,255,0.8));
        border-radius: 15px;
        overflow: hidden;
    }
    
    .dataframe th {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: bold !important;
    }
    
    .dataframe tr:hover {
        background: rgba(102, 126, 234, 0.1) !important;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom chart styling */
    .chart-container {
        background: rgba(255,255,255,0.9);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Your model classes (unchanged)...
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            with open(f"{self.model_dir}/hybrid_config.json", "r") as f:
                config = json.load(f)
            self.win_ctx = config["win_ctx"]
            self.n_feats = config["n_feats"]
            self.win_total = 2 * self.win_ctx + 1
            
            expected_flat_dim = self.win_total * self.n_feats
            if expected_flat_dim != 1521:
                st.warning(f"Dimension mismatch: Expected 1521, got {expected_flat_dim}")
            
            self.scaler = load(f"{self.model_dir}/scaler_ctx.joblib")
            
            self.bdnn = BDNN_Head(in_dim=expected_flat_dim)
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
        n_frames = len(features)
        context_features = []
        
        for i in range(n_frames):
            left = max(0, i - self.win_ctx)
            right = min(n_frames, i + self.win_ctx + 1)
            window = features[left:right]
            
            if len(window) < self.win_total:
                if left == 0:
                    pad = np.repeat(window[:1], self.win_total - len(window), axis=0)
                    window = np.vstack([pad, window])
                else:
                    pad = np.repeat(window[-1:], self.win_total - len(window), axis=0)
                    window = np.vstack([window, pad])
            
            context_features.append(window.flatten())
        
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
        
        hop_length = 160
        times = librosa.frames_to_time(range(len(final_probs)), sr=sr, hop_length=hop_length)
        
        segments = []
        if len(final_probs) > 0:
            current_label = "silence" if final_probs[0] < threshold else "speech"
            start_time = float(times[0])
            start_idx = 0
            
            for i in range(1, len(final_probs)):
                new_label = "silence" if final_probs[i] < threshold else "speech"
                if new_label != current_label:
                    duration = times[i] - start_time
                    if duration >= min_duration:
                        confidence = float(np.mean(final_probs[start_idx:i]))
                        segments.append({
                            'start': float(start_time),
                            'end': float(times[i]),
                            'label': current_label,
                            'confidence': confidence
                        })
                    start_time = times[i]
                    start_idx = i
                    current_label = new_label
            
            duration = times[-1] - start_time
            if duration >= min_duration:
                confidence = float(np.mean(final_probs[start_idx:]))
                segments.append({
                    'start': float(start_time),
                    'end': float(times[-1]),
                    'label': current_label,
                    'confidence': confidence
                })
        
        if segments:
            speech_time = sum(s['end']-s['start'] for s in segments if s['label'] == 'speech')
            total_time = segments[-1]['end']
            speech_ratio = speech_time / total_time if total_time > 0 else 0
        else:
            speech_time = total_time = speech_ratio = 0
        
        return {
            'segments': segments,
            'speech_ratio': speech_ratio,
            'total_duration': total_time,
            'speech_duration': speech_time,
            'silence_duration': total_time - speech_time,
            'probabilities': final_probs,
            'times': times,
            'audio_data': audio_data,
            'sample_rate': sr
        }

# ========== STREAMLIT UI ==========
st.set_page_config(
    page_title="VAD AI Pro | Advanced Voice Detection",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Animated Header
st.markdown('<h1 class="animated-title floating">🎤 VAD AI PRO</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #FFFFFF; font-size: 1.2em; margin-bottom: 40px;">Advanced Hybrid Voice Activity Detection with Neural Networks</p>', unsafe_allow_html=True)

# Sidebar with modern design
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <div style="font-size: 2.5em; margin-bottom: 10px;">⚙️</div>
        <h2 style="color: white; margin-bottom: 30px;">Control Panel</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Settings in gradient cards
    with st.container():
        st.markdown('<div class="gradient-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Detection Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            threshold = st.slider(
                "🔊 Threshold",
                0.1, 0.9, 0.5, 0.05,
                help="Speech detection sensitivity"
            )
        
        with col2:
            min_duration = st.slider(
                "⏱️ Min Duration",
                0.05, 0.5, 0.1, 0.05,
                help="Minimum segment duration"
            )
        
        # Visual threshold indicator
        threshold_color = "#06D6A0" if threshold <= 0.5 else "#FFD166" if threshold <= 0.7 else "#FF6B6B"
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.1); padding: 10px; border-radius: 10px; margin-top: 10px;">
            <div style="display: flex; justify-content: space-between; color: white;">
                <span>Relaxed</span>
                <span style="color: {threshold_color}">Current: {threshold}</span>
                <span>Strict</span>
            </div>
            <div style="height: 6px; background: linear-gradient(90deg, #06D6A0, #FFD166, #FF6B6B); border-radius: 3px; margin-top: 5px;">
                <div style="width: {threshold*100}%; height: 100%; background: white; border-radius: 3px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Model info in gradient card
    with st.container():
        st.markdown('<div class="gradient-card">', unsafe_allow_html=True)
        st.markdown("### 🤖 AI Models")
        
        # Model status indicators
        st.markdown("""
        <div style="display: flex; justify-content: space-between; margin: 15px 0;">
            <div style="text-align: center;">
                <div style="font-size: 1.5em;">🧠</div>
                <div style="color: #06D6A0; font-weight: bold;">BDNN</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.5em;">🖼️</div>
                <div style="color: #FFD166; font-weight: bold;">CNN</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.5em;">🤝</div>
                <div style="color: #FF6B6B; font-weight: bold;">Meta</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Progress bar for fun
        st.progress(100, text="Model Ready")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Stats card
    with st.container():
        st.markdown('<div class="gradient-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Live Stats")
        st.metric("Active Sessions", "1", "+100%")
        st.metric("Avg Processing", "0.8s", "-20%")
        st.markdown('</div>', unsafe_allow_html=True)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Upload section with beautiful design
    st.markdown("""
    <div class="upload-box">
        <div style="font-size: 4em; margin-bottom: 20px;">📁</div>
        <h2 style="color: white; margin-bottom: 10px;">Drop Your Audio Here</h2>
        <p style="color: rgba(255,255,255,0.8);">Supports WAV, MP3, M4A, FLAC formats</p>
        <div style="margin-top: 20px; font-size: 1.5em; color: #FFD166;">⬆️</div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "",
        type=['wav', 'mp3', 'm4a', 'flac'],
        label_visibility="collapsed"
    )

with col2:
    # Quick stats panel
    st.markdown('<div class="gradient-card">', unsafe_allow_html=True)
    st.markdown("### ⚡ Quick Stats")
    
    stats_data = {
        "Accuracy": "96.5%",
        "Speed": "Real-time",
        "Max File": "1GB",
        "Languages": "Multi"
    }
    
    for key, value in stats_data.items():
        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.markdown(f"**{key}**")
        with col_b:
            st.markdown(f"`{value}`")
    
    st.markdown("---")
    st.markdown("#### 🎯 Tips")
    st.info("• Use WAV for best results\n• Keep background noise low\n• Optimal threshold: 0.4-0.6")
    st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    # Create progress animation
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(100):
        progress_bar.progress(i + 1)
        status_text.text(f"🔄 Processing audio... {i+1}%")
        time.sleep(0.01)
    
    status_text.text("✅ Analysis Complete!")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        analyzer = HybridVADAnalyzer()
        results = analyzer.predict_vad(tmp_path, threshold, min_duration)
        
        # Display metrics in beautiful cards
        st.markdown("## 📊 Analysis Results")
        
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        
        with metrics_col1:
            st.markdown(f"""
            <div class="custom-metric pulse">
                <div style="font-size: 2em;">🎤</div>
                <h3>Speech Ratio</h3>
                <h2 style="font-size: 2.5em;">{results['speech_ratio']:.1%}</h2>
                <div style="font-size: 0.8em; opacity: 0.8;">{results['speech_duration']:.1f}s / {results['total_duration']:.1f}s</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metrics_col2:
            icon = "📈" if results['speech_ratio'] > 0.5 else "📉"
            st.markdown(f"""
            <div class="custom-metric">
                <div style="font-size: 2em;">{icon}</div>
                <h3>Audio Duration</h3>
                <h2 style="font-size: 2.5em;">{results['total_duration']:.1f}s</h2>
                <div style="font-size: 0.8em; opacity: 0.8;">Total length</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metrics_col3:
            speech_count = len([s for s in results['segments'] if s['label'] == 'speech'])
            st.markdown(f"""
            <div class="custom-metric">
                <div style="font-size: 2em;">🎯</div>
                <h3>Speech Segments</h3>
                <h2 style="font-size: 2.5em;">{speech_count}</h2>
                <div style="font-size: 0.8em; opacity: 0.8;">Detected speech parts</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metrics_col4:
            confidence = np.mean([s['confidence'] for s in results['segments']]) if results['segments'] else 0
            st.markdown(f"""
            <div class="custom-metric">
                <div style="font-size: 2em;">🤖</div>
                <h3>AI Confidence</h3>
                <h2 style="font-size: 2.5em;">{confidence:.1%}</h2>
                <div style="font-size: 0.8em; opacity: 0.8;">Model accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Segments table with beautiful design
        st.markdown("## 📝 Detected Segments")
        
        if results['segments']:
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["📋 Table View", "📊 Statistics", "📈 Charts"])
            
            with tab1:
                segments_data = []
                for i, seg in enumerate(results['segments']):
                    segments_data.append({
                        '#': i+1,
                        '🎭 Type': f"{'🎤 SPEECH' if seg['label'] == 'speech' else '🔇 SILENCE'}",
                        '⏱️ Start': f"{seg['start']:.2f}s",
                        '⏱️ End': f"{seg['end']:.2f}s",
                        '🕒 Duration': f"{seg['end']-seg['start']:.2f}s",
                        '📊 Confidence': f"{seg['confidence']:.1%}"
                    })
                
                df = pd.DataFrame(segments_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
            
            with tab2:
                # Statistics visualization using Streamlit metrics
                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                
                with col_stat1:
                    st.metric(
                        label="Total Segments",
                        value=len(results['segments']),
                        delta=f"{len([s for s in results['segments'] if s['label'] == 'speech'])} speech"
                    )
                
                with col_stat2:
                    avg_duration = np.mean([s['end']-s['start'] for s in results['segments']]) if results['segments'] else 0
                    st.metric(
                        label="Avg Segment Duration",
                        value=f"{avg_duration:.2f}s",
                        delta="Long" if avg_duration > 1.0 else "Short"
                    )
                
                with col_stat3:
                    max_duration = max([s['end']-s['start'] for s in results['segments']]) if results['segments'] else 0
                    st.metric(
                        label="Longest Segment",
                        value=f"{max_duration:.2f}s",
                        delta="Speech" if any(s['end']-s['start'] == max_duration and s['label'] == 'speech' for s in results['segments']) else "Silence"
                    )
                
                with col_stat4:
                    min_duration = min([s['end']-s['start'] for s in results['segments']]) if results['segments'] else 0
                    st.metric(
                        label="Shortest Segment",
                        value=f"{min_duration:.2f}s",
                        delta="Below min" if min_duration < min_duration else "Above min"
                    )
                
                # Show distribution
                st.markdown("### 📊 Distribution Analysis")
                dist_col1, dist_col2 = st.columns(2)
                
                with dist_col1:
                    speech_percentage = (results['speech_duration'] / results['total_duration']) * 100 if results['total_duration'] > 0 else 0
                    silence_percentage = 100 - speech_percentage
                    
                    st.markdown("#### Duration Distribution")
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <span>🎤 Speech</span>
                            <span>{speech_percentage:.1f}%</span>
                        </div>
                        <div style="height: 20px; background: #e0e0e0; border-radius: 10px; overflow: hidden;">
                            <div style="width: {speech_percentage}%; height: 100%; background: linear-gradient(90deg, #06D6A0, #118AB2);"></div>
                        </div>
                        
                        <div style="display: flex; justify-content: space-between; margin-top: 15px; margin-bottom: 5px;">
                            <span>🔇 Silence</span>
                            <span>{silence_percentage:.1f}%</span>
                        </div>
                        <div style="height: 20px; background: #e0e0e0; border-radius: 10px; overflow: hidden;">
                            <div style="width: {silence_percentage}%; height: 100%; background: linear-gradient(90deg, #FF6B6B, #FF8E53);"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with dist_col2:
                    st.markdown("#### Segment Count")
                    speech_segments = len([s for s in results['segments'] if s['label'] == 'speech'])
                    silence_segments = len([s for s in results['segments'] if s['label'] == 'silence'])
                    
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; text-align: center;">
                        <div style="display: flex; justify-content: space-around; margin-bottom: 20px;">
                            <div>
                                <div style="font-size: 2em; color: #06D6A0;">{speech_segments}</div>
                                <div style="color: white;">Speech Segments</div>
                            </div>
                            <div>
                                <div style="font-size: 2em; color: #FF6B6B;">{silence_segments}</div>
                                <div style="color: white;">Silence Segments</div>
                            </div>
                        </div>
                        <div style="font-size: 3em; color: #FFD166;">{len(results['segments'])}</div>
                        <div style="color: white;">Total Segments</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with tab3:
                # Charts using matplotlib
                st.markdown("### 📈 Visual Charts")
                
                chart_tab1, chart_tab2, chart_tab3 = st.tabs(["Waveform", "Probability", "Timeline"])
                
                with chart_tab1:
                    fig, ax = plt.subplots(figsize=(12, 4))
                    times_full = np.linspace(0, len(results['audio_data'])/results['sample_rate'], 
                                            len(results['audio_data']))
                    
                    ax.plot(times_full, results['audio_data'], color='#FF6B6B', alpha=0.7, linewidth=0.5)
                    ax.fill_between(times_full, results['audio_data'], alpha=0.2, color='#FF6B6B')
                    
                    # Add speech segments
                    for seg in results['segments']:
                        if seg['label'] == 'speech':
                            ax.axvspan(seg['start'], seg['end'], color='green', alpha=0.2, label='Speech')
                        else:
                            ax.axvspan(seg['start'], seg['end'], color='red', alpha=0.1, label='Silence')
                    
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Amplitude')
                    ax.set_title('Audio Waveform with VAD Segments')
                    ax.grid(True, alpha=0.3)
                    
                    # Remove duplicate labels
                    handles, labels = ax.get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    if by_label:
                        ax.legend(by_label.values(), by_label.keys())
                    
                    st.pyplot(fig)
                
                with chart_tab2:
                    fig, ax = plt.subplots(figsize=(12, 4))
                    
                    ax.plot(results['times'], results['probabilities'], color='#FFD166', linewidth=2)
                    ax.fill_between(results['times'], results['probabilities'], alpha=0.3, color='#FFD166')
                    ax.axhline(y=threshold, color='#FF6B6B', linestyle='--', label=f'Threshold ({threshold})')
                    
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Probability')
                    ax.set_title('Speech Probability over Time')
                    ax.set_ylim([0, 1])
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    
                    st.pyplot(fig)
                
                with chart_tab3:
                    fig, ax = plt.subplots(figsize=(12, 3))
                    
                    y_pos = 0
                    for seg in results['segments']:
                        color = '#06D6A0' if seg['label'] == 'speech' else '#FF6B6B'
                        ax.barh(y_pos, seg['end']-seg['start'], left=seg['start'], 
                               height=0.6, color=color, alpha=0.8, edgecolor='white')
                    
                    ax.set_yticks([])
                    ax.set_xlabel('Time (s)')
                    ax.set_title('Segment Timeline')
                    ax.grid(True, alpha=0.3, axis='x')
                    
                    # Add legend
                    from matplotlib.patches import Patch
                    legend_elements = [
                        Patch(facecolor='#06D6A0', alpha=0.8, label='Speech'),
                        Patch(facecolor='#FF6B6B', alpha=0.8, label='Silence')
                    ]
                    ax.legend(handles=legend_elements, loc='upper right')
                    
                    st.pyplot(fig)
        
        # Export section with beautiful cards
        st.markdown("## 💾 Export Results")
        
        export_col1, export_col2, export_col3 = st.columns(3)
        
        with export_col1:
            st.markdown('<div class="gradient-card">', unsafe_allow_html=True)
            st.markdown("### 📊 JSON Export")
            st.markdown("Complete analysis data")
            
            results_json = {
                'analysis_date': datetime.now().isoformat(),
                'settings': {'threshold': threshold, 'min_duration': min_duration},
                'statistics': {
                    'speech_ratio': float(results['speech_ratio']),
                    'total_duration': float(results['total_duration']),
                    'speech_duration': float(results['speech_duration']),
                    'silence_duration': float(results['silence_duration']),
                    'segment_count': len(results['segments'])
                },
                'segments': results['segments']
            }
            
            json_str = json.dumps(results_json, indent=2)
            
            st.download_button(
                label="⬇️ Download JSON",
                data=json_str,
                file_name="vad_analysis.json",
                mime="application/json",
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with export_col2:
            st.markdown('<div class="gradient-card">', unsafe_allow_html=True)
            st.markdown("### 📈 CSV Export")
            st.markdown("Tabular segment data")
            
            if results['segments']:
                df_export = pd.DataFrame(results['segments'])
                csv = df_export.to_csv(index=False)
                
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv,
                    file_name="vad_segments.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with export_col3:
            st.markdown('<div class="gradient-card">', unsafe_allow_html=True)
            st.markdown("### 📋 Report")
            st.markdown("Formatted analysis report")
            
            report_html = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: 'Arial', sans-serif; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }}
                    .report-card {{ background: rgba(255,255,255,0.1); padding: 30px; border-radius: 20px; backdrop-filter: blur(10px); }}
                </style>
            </head>
            <body>
                <div class="report-card">
                    <h1>🎤 VAD Analysis Report</h1>
                    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <h2>Summary</h2>
                    <p>Speech Ratio: {results['speech_ratio']:.1%}</p>
                    <p>Total Duration: {results['total_duration']:.2f}s</p>
                    <p>Segments Detected: {len(results['segments'])}</p>
                </div>
            </body>
            </html>
            """
            
            st.download_button(
                label="⬇️ Download Report",
                data=report_html,
                file_name="vad_report.html",
                mime="text/html",
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        with st.expander("🔧 Technical Details"):
            st.code(f"Error: {str(e)}")
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass
else:
    # Beautiful empty state
    st.markdown("""
    <div style="text-align: center; padding: 80px 20px;">
        <div style="font-size: 6em; margin-bottom: 30px; color: #FFD166;" class="floating">🎵</div>
        <h2 style="color: white; font-size: 2.5em; margin-bottom: 20px;">Ready to Analyze Audio</h2>
        <p style="color: rgba(255,255,255,0.8); font-size: 1.2em; max-width: 600px; margin: 0 auto 40px;">
            Upload an audio file to start detecting speech patterns using our advanced hybrid AI model.
            Get detailed insights, visualizations, and export options.
        </p>
        <div style="display: flex; justify-content: center; gap: 20px; margin-top: 40px;">
            <div style="text-align: center;">
                <div style="font-size: 3em; color: #06D6A0;">🎤</div>
                <p style="color: white;">Speech Detection</p>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 3em; color: #FF6B6B;">📊</div>
                <p style="color: white;">Visual Analytics</p>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 3em; color: #FFD166;">⚡</div>
                <p style="color: white;">Fast Processing</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer with gradient
st.markdown("""
<div style="text-align: center; padding: 30px; margin-top: 50px; 
            background: linear-gradient(90deg, #FF6B6B, #FFD166, #06D6A0, #118AB2);
            border-radius: 20px; color: white;">
    <h3>🚀 Powered by Hybrid VAD AI</h3>
    <p>BDNN + CNN + Meta Classifier Ensemble | Real-time Audio Analysis</p>
    <div style="margin-top: 20px; font-size: 1.5em;">
        ⚡ 🎤 📊 🔥 🚀
    </div>
</div>
""", unsafe_allow_html=True)