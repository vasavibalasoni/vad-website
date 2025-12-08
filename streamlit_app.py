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
from matplotlib.patches import Patch

# Fix for deployment
os.environ['MPLCONFIGDIR'] = '/tmp/.matplotlib'

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Your EXACT model classes from training
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
            
            # Load config
            with open(f"{self.model_dir}/hybrid_config.json", "r") as f:
                config = json.load(f)
            self.win_ctx = config["win_ctx"]  # Should be 19
            self.n_feats = config["n_feats"]  # Should be 39
            self.win_total = 2 * self.win_ctx + 1  # Should be 39
            
            # Verify dimensions
            expected_flat_dim = self.win_total * self.n_feats
            if expected_flat_dim != 1521:
                st.warning(f"Dimension mismatch: Expected 1521, got {expected_flat_dim}")
            
            # Load scaler and models
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
        # EXACTLY as in training: 13 MFCC + delta + delta2 = 39 features
        y, sr = librosa.load(audio_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=160)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        features = np.vstack([mfcc, delta, delta2]).T  # Shape: (n_frames, 39)
        return features, sr, y

    def add_context(self, features):
        """EXACTLY matches training code's context windowing"""
        n_frames = len(features)
        context_features = []
        
        for i in range(n_frames):
            left = max(0, i - self.win_ctx)
            right = min(n_frames, i + self.win_ctx + 1)
            window = features[left:right]
            
            # Pad if needed (same logic as training)
            if len(window) < self.win_total:
                if left == 0:
                    # Pad at beginning
                    pad = np.repeat(window[:1], self.win_total - len(window), axis=0)
                    window = np.vstack([pad, window])
                else:
                    # Pad at end
                    pad = np.repeat(window[-1:], self.win_total - len(window), axis=0)
                    window = np.vstack([window, pad])
            
            context_features.append(window.flatten())
        
        return np.array(context_features)

    def predict_vad(self, audio_path, threshold=0.5, min_duration=0.1):
        features, sr, audio_data = self.extract_features(audio_path)
        features_ctx = self.add_context(features)
        
        # Scale features
        features_scaled = self.scaler.transform(features_ctx)
        
        # Convert to torch tensors
        features_flat = torch.tensor(features_scaled, dtype=torch.float32).to(self.device)
        features_cnn = features_flat.reshape(-1, 1, self.win_total, self.n_feats)
        
        # Predict
        with torch.no_grad():
            bdnn_probs = self.bdnn(features_flat).cpu().numpy()
            cnn_probs = self.cnn(features_cnn).cpu().numpy()
            meta_input = torch.tensor(np.column_stack([bdnn_probs, cnn_probs]), dtype=torch.float32).to(self.device)
            final_probs = self.meta(meta_input).cpu().numpy()
        
        # Create time array
        hop_length = 160
        times = librosa.frames_to_time(range(len(final_probs)), sr=sr, hop_length=hop_length)
        
        # ✅ FIXED: Segment merging logic to match Colab output
        segments = []
        if len(final_probs) == 0:
            return {
                'segments': segments,
                'speech_ratio': 0,
                'total_duration': 0,
                'speech_duration': 0,
                'silence_duration': 0,
                'probabilities': final_probs,
                'times': times,
                'audio_data': audio_data,
                'sample_rate': sr
            }
        
        # Create binary speech/silence labels
        is_speech = final_probs >= threshold
        
        # Initialize first segment
        current_start = 0
        current_label = "speech" if is_speech[0] else "silence"
        
        # Iterate through all frames
        for i in range(1, len(is_speech)):
            if is_speech[i] != is_speech[i-1]:
                # Segment boundary found
                segment_end = i
                duration = times[segment_end] - times[current_start]
                
                # Only add segment if it meets minimum duration
                if duration >= min_duration:
                    # Calculate confidence as mean probability in this segment
                    segment_probs = final_probs[current_start:segment_end]
                    confidence = float(np.mean(segment_probs))
                    
                    segments.append({
                        'start': float(times[current_start]),
                        'end': float(times[segment_end]),
                        'label': current_label,
                        'confidence': confidence
                    })
                
                # Start new segment
                current_start = segment_end
                current_label = "speech" if is_speech[i] else "silence"
        
        # Handle the last segment
        duration = times[-1] - times[current_start]
        if duration >= min_duration:
            segment_probs = final_probs[current_start:]
            confidence = float(np.mean(segment_probs))
            
            segments.append({
                'start': float(times[current_start]),
                'end': float(times[-1]),
                'label': current_label,
                'confidence': confidence
            })
        
        # Statistics
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

# Streamlit UI
st.set_page_config(page_title="Hybrid VAD Analyzer", page_icon="🎵", layout="wide")

st.title("🎵 Hybrid VAD Analyzer")
st.markdown("Upload an audio file to detect speech and silence segments")

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Detection Threshold", 0.1, 0.9, 0.5, 0.05)
    min_duration = st.slider("Minimum Segment Duration (s)", 0.05, 0.5, 0.1, 0.05)
    
    st.header("Model Info")
    st.info("Using Hybrid Model: BDNN + CNN + Meta Classifier")

uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'm4a', 'flac'])

if uploaded_file is not None:
    with st.spinner("Analyzing audio..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
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
            st.subheader("📊 Detected Segments")
            
            if results['segments']:
                # Create a dataframe for better display
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
                
                # Display as table
                df = pd.DataFrame(segments_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Display detailed segments
                with st.expander("📝 View Detailed Segment Information"):
                    for i, seg in enumerate(results['segments']):
                        color = "🟢" if seg['label'] == 'speech' else "🔴"
                        st.write(f"{color} **{seg['label'].upper()}** | "
                                f"Time: {seg['start']:.2f}s - {seg['end']:.2f}s | "
                                f"Duration: {seg['end']-seg['start']:.2f}s | "
                                f"Confidence: {seg['confidence']:.1%}")
            else:
                st.warning("No segments detected with current settings. Try lowering the threshold or minimum duration.")
            
            # Visualization
            st.subheader("📈 Visualization")
            
            # Create tabs for different visualizations
            tab1, tab2, tab3 = st.tabs(["Waveform with Segments", "Probability Plot", "Segment Timeline"])
            
            with tab1:
                # Waveform with segments
                fig1, ax1 = plt.subplots(figsize=(12, 3))
                times_full = np.linspace(0, len(results['audio_data'])/results['sample_rate'], len(results['audio_data']))
                ax1.plot(times_full, results['audio_data'], alpha=0.7, color='blue', linewidth=0.5)
                
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
                ax1.grid(True, alpha=0.3)
                st.pyplot(fig1)
            
            with tab2:
                # Probability plot
                fig2, ax2 = plt.subplots(figsize=(12, 3))
                ax2.plot(results['times'], results['probabilities'], 'b-', alpha=0.7, linewidth=1, label='Speech Probability')
                ax2.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, label=f'Threshold ({threshold})')
                
                # Fill area under curve
                ax2.fill_between(results['times'], 0, results['probabilities'], alpha=0.3)
                
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
            
            with tab3:
                # Segment timeline visualization
                fig3, ax3 = plt.subplots(figsize=(12, 2))
                
                y_pos = 0
                for seg in results['segments']:
                    color = 'green' if seg['label'] == 'speech' else 'red'
                    ax3.barh(y_pos, seg['end']-seg['start'], left=seg['start'], 
                            height=0.8, color=color, alpha=0.7, edgecolor='black')
                
                ax3.set_yticks([])
                ax3.set_xlabel("Time (s)")
                ax3.set_title("Segment Timeline")
                ax3.grid(True, alpha=0.3, axis='x')
                
                # Add legend
                legend_elements = [Patch(facecolor='green', alpha=0.7, label='Speech'),
                                 Patch(facecolor='red', alpha=0.7, label='Silence')]
                ax3.legend(handles=legend_elements, loc='upper right')
                
                st.pyplot(fig3)
            
            # Export results
            st.subheader("💾 Export Results")
            
            # Create JSON for download
            results_json = {
                'settings': {
                    'threshold': threshold,
                    'min_duration': min_duration
                },
                'statistics': {
                    'speech_ratio': float(results['speech_ratio']),
                    'total_duration': float(results['total_duration']),
                    'speech_duration': float(results['speech_duration']),
                    'silence_duration': float(results['silence_duration'])
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
                if results['segments']:
                    df_export = pd.DataFrame(results['segments'])
                    csv = df_export.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv,
                        file_name="vad_results.csv",
                        mime="text/csv"
                    )
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            st.error("Please ensure:")
            st.error("1. Your model files are in the 'hybrid_model' directory")
            st.error("2. The directory contains: hybrid_config.json, scaler_ctx.joblib, bdnn.pth, cnn.pth, meta.pth")
        finally:
            # Clean up
            try:
                os.unlink(tmp_path)
            except:
                pass
else:
    # Show instructions when no file is uploaded
    st.info("👆 Please upload an audio file to begin analysis.")
    
    with st.expander("🎯 How to Use"):
        st.markdown("""
        1. **Upload** an audio file (WAV, MP3, M4A, or FLAC)
        2. **Adjust settings** in the sidebar:
           - **Detection Threshold**: Higher = more strict speech detection
           - **Minimum Duration**: Filter out very short segments
        3. **View results**:
           - Statistics at the top
           - Segment table with timestamps
           - Visualizations of the waveform
        4. **Export** results as JSON or CSV
        """)

st.markdown("---")
st.markdown("### 🚀 Powered by Hybrid VAD Model | Built with Streamlit")