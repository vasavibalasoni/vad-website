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

# Fix for deployment
os.environ['MPLCONFIGDIR'] = '/tmp/.matplotlib'

# Suppress sklearn version warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Your model classes
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
        # Suppress warnings during model loading
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            with open(f"{self.model_dir}/hybrid_config.json", "r") as f:
                config = json.load(f)
            self.win_ctx = config["win_ctx"]
            self.n_feats = config["n_feats"]
            self.win_total = 2 * self.win_ctx + 1
            self.flat_dim = self.win_total * self.n_feats
            
            # Use the pre-trained scaler
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
        
        # Use pre-trained scaler
        features_scaled = self.scaler.transform(features_ctx)
        
        features_flat = torch.tensor(features_scaled, dtype=torch.float32).to(self.device)
        features_cnn = features_flat.reshape(-1, 1, self.win_total, self.n_feats)
        
        with torch.no_grad():
            bdnn_probs = self.bdnn(features_flat).cpu().numpy()
            cnn_probs = self.cnn(features_cnn).cpu().numpy()
            meta_input = torch.tensor(np.column_stack([bdnn_probs, cnn_probs]), dtype=torch.float32).to(self.device)
            final_probs = self.meta(meta_input).cpu().numpy()
        
        # Calculate time array correctly
        hop_length = 160
        times = librosa.frames_to_time(range(len(final_probs)), sr=sr, hop_length=hop_length)
        
        segments = []
        if len(final_probs) > 0:
            current_label = "silence" if final_probs[0] < threshold else "speech"
            start_time = float(times[0])
            start_idx = 0
            
            # First pass: Collect all potential segment boundaries
            boundaries = []
            for i in range(1, len(final_probs)):
                new_label = "silence" if final_probs[i] < threshold else "speech"
                if new_label != current_label:
                    boundaries.append(i)
                    current_label = new_label
            
            # Second pass: Merge short segments
            boundaries.append(len(final_probs))  # Add end boundary
            seg_start_idx = 0
            
            for i, boundary in enumerate(boundaries):
                seg_end_idx = boundary
                seg_start_time = times[seg_start_idx]
                seg_end_time = times[seg_end_idx]
                duration = seg_end_time - seg_start_time
                
                # Get label for this segment
                seg_label = "silence" if final_probs[seg_start_idx] < threshold else "speech"
                
                # Only create segment if duration is sufficient
                if duration >= min_duration:
                    # Calculate confidence using 100 fps conversion like Colab
                    start_frame = int(seg_start_time * 100)
                    end_frame = int(seg_end_time * 100)
                    start_frame = min(start_frame, len(final_probs))
                    end_frame = min(end_frame, len(final_probs))
                    segment_probs = final_probs[start_frame:end_frame]
                    confidence = float(np.mean(segment_probs)) if len(segment_probs) > 0 else 0.0
                    
                    segments.append({
                        'start': float(seg_start_time),
                        'end': float(seg_end_time),
                        'label': seg_label,
                        'confidence': confidence
                    })
                    seg_start_idx = seg_end_idx
            
            # Handle the final segment
            if seg_start_idx < len(final_probs):
                seg_start_time = times[seg_start_idx]
                seg_end_time = times[-1]
                duration = seg_end_time - seg_start_time
                
                if duration >= min_duration:
                    seg_label = "silence" if final_probs[seg_start_idx] < threshold else "speech"
                    start_frame = int(seg_start_time * 100)
                    end_frame = int(seg_end_time * 100)
                    start_frame = min(start_frame, len(final_probs))
                    end_frame = min(end_frame, len(final_probs))
                    segment_probs = final_probs[start_frame:end_frame]
                    confidence = float(np.mean(segment_probs)) if len(segment_probs) > 0 else 0.0
                    
                    segments.append({
                        'start': float(seg_start_time),
                        'end': float(seg_end_time),
                        'label': seg_label,
                        'confidence': confidence
                    })
        
        # Calculate statistics
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
            'times': times
        }

# Streamlit UI
st.set_page_config(page_title="Hybrid VAD Analyzer", page_icon="🎵", layout="wide")

st.title("🎵 Hybrid VAD Analyzer")
st.markdown("Upload an audio file to detect speech and silence segments using advanced AI")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Detection Threshold", 0.1, 0.9, 0.5, 0.05)
    min_duration = st.slider("Minimum Segment Duration (s)", 0.05, 0.5, 0.15, 0.05)  # Changed default to 0.15s
    
    st.header("About")
    st.markdown("""
    This app uses a hybrid AI model combining:
    - BDNN (Bidirectional Deep Neural Network)
    - CNN (Convolutional Neural Network)
    - Meta Classifier (for final decision)
    """)

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
            
            # Display statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Speech Ratio", f"{results['speech_ratio']:.1%}")
            with col2:
                st.metric("Total Duration", f"{results['total_duration']:.2f}s")
            with col3:
                st.metric("Speech Duration", f"{results['speech_duration']:.2f}s")
            with col4:
                st.metric("Silence Duration", f"{results['silence_duration']:.2f}s")
            
            # Display segments in an expandable table
            st.subheader("Detected Segments")
            
            # Create a table view
            if results['segments']:
                segments_data = []
                for i, seg in enumerate(results['segments']):
                    segments_data.append({
                        'Segment': i+1,
                        'Type': '🎤 SPEECH' if seg['label'] == 'speech' else '🔇 SILENCE',
                        'Start (s)': f"{seg['start']:.2f}",
                        'End (s)': f"{seg['end']:.2f}",
                        'Duration (s)': f"{seg['end']-seg['start']:.2f}",
                        'Confidence': f"{seg['confidence']:.1%}"
                    })
                
                st.dataframe(segments_data, use_container_width=True)
                
                # Show detailed view
                with st.expander("View Detailed Segment Information"):
                    for i, seg in enumerate(results['segments']):
                        color = "🟢" if seg['label'] == 'speech' else "🔴"
                        st.write(f"{color} **{seg['label'].upper()}** | "
                                f"Time: {seg['start']:.2f}s - {seg['end']:.2f}s | "
                                f"Duration: {seg['end']-seg['start']:.2f}s | "
                                f"Confidence: {seg['confidence']:.1%}")
            else:
                st.info("No segments detected with current settings.")
            
            # Visualization
            st.subheader("Visualization")
            
            # Create two tabs for different visualizations
            tab1, tab2 = st.tabs(["Waveform with Segments", "Probability Plot"])
            
            with tab1:
                # Waveform with segments
                y, sr = librosa.load(tmp_path)
                fig1, ax1 = plt.subplots(figsize=(12, 3))
                librosa.display.waveshow(y, sr=sr, alpha=0.7, color='blue', ax=ax1)
                
                for seg in results['segments']:
                    color = 'green' if seg['label'] == 'speech' else 'red'
                    ax1.axvspan(seg['start'], seg['end'], color=color, alpha=0.3, label=seg['label'].capitalize())
                
                # Remove duplicate labels
                handles, labels = ax1.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                if by_label:
                    ax1.legend(by_label.values(), by_label.keys())
                
                ax1.set_title("Voice Activity Detection Results")
                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("Amplitude")
                st.pyplot(fig1)
            
            with tab2:
                # Probability plot
                fig2, ax2 = plt.subplots(figsize=(12, 3))
                ax2.plot(results['times'], results['probabilities'], 'b-', alpha=0.7, label='Speech Probability')
                ax2.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, label=f'Threshold ({threshold})')
                
                # Highlight segments
                for seg in results['segments']:
                    if seg['label'] == 'speech':
                        ax2.axvspan(seg['start'], seg['end'], color='green', alpha=0.2)
                    else:
                        ax2.axvspan(seg['start'], seg['end'], color='red', alpha=0.2)
                
                ax2.set_title("Speech Probability over Time")
                ax2.set_xlabel("Time (s)")
                ax2.set_ylabel("Probability")
                ax2.set_ylim([0, 1])
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                st.pyplot(fig2)
            
            # Download results
            st.subheader("Export Results")
            
            # Create JSON for download
            results_json = {
                'settings': {
                    'threshold': threshold,
                    'min_duration': min_duration
                },
                'statistics': {
                    'speech_ratio': results['speech_ratio'],
                    'total_duration': results['total_duration'],
                    'speech_duration': results['speech_duration'],
                    'silence_duration': results['silence_duration']
                },
                'segments': results['segments']
            }
            
            import json as json_module
            json_str = json_module.dumps(results_json, indent=2)
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📥 Download Results (JSON)",
                    data=json_str,
                    file_name="vad_results.json",
                    mime="application/json"
                )
            
            with col2:
                # Create CSV
                import pandas as pd
                if results['segments']:
                    df = pd.DataFrame(results['segments'])
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv,
                        file_name="vad_results.csv",
                        mime="text/csv"
                    )
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            st.error("Please ensure your model files are in the 'hybrid_model' directory.")
        finally:
            # Clean up
            try:
                os.unlink(tmp_path)
            except:
                pass

else:
    # Show example when no file is uploaded
    st.info("👆 Please upload an audio file to begin analysis.")
    
    # Show example structure
    with st.expander("ℹ️ Expected Model File Structure"):
        st.code("""
your_app_directory/
├── streamlit_app.py
├── requirements.txt
├── README.md
└── hybrid_model/
    ├── hybrid_config.json
    ├── scaler_ctx.joblib
    ├── bdnn.pth
    ├── cnn.pth
    └── meta.pth
        """)

st.markdown("---")
st.markdown("### 🚀 Powered by Hybrid VAD Model | Built with Streamlit")