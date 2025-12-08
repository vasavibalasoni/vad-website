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
            'times': times,
            'audio_data': audio_data,
            'sample_rate': sr
        }

# Streamlit UI
st.set_page_config(page_title="Hybrid VAD Analyzer", page_icon="🎵", layout="wide")

st.title("🎵 Hybrid VAD Analyzer")
st.markdown("Upload an audio file to detect speech and silence segments using advanced AI")

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Detection Threshold", 0.1, 0.9, 0.5, 0.05,
                         help="Higher values = more strict speech detection")
    min_duration = st.slider("Minimum Segment Duration (s)", 0.05, 0.5, 0.1, 0.05,
                           help="Filter out segments shorter than this duration")
    
    st.header("Model Info")
    st.info("""
    **Hybrid Model Architecture:**
    - BDNN (Bidirectional DNN): Temporal context
    - CNN: Spatial feature extraction  
    - Meta Classifier: Final fusion
    """)

uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'm4a', 'flac'],
                                 help="Supported formats: WAV, MP3, M4A, FLAC")

if uploaded_file is not None:
    with st.spinner("Analyzing audio..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Initialize analyzer
            analyzer = HybridVADAnalyzer()
            
            # Show model info in sidebar
            with st.sidebar:
                st.success(f"✓ Model loaded: win_ctx={analyzer.win_ctx}, n_feats={analyzer.n_feats}")
            
            # Run prediction
            results = analyzer.predict_vad(tmp_path, threshold, min_duration)
            
            # Display statistics
            st.subheader("📊 Analysis Results")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Speech Ratio", f"{results['speech_ratio']:.1%}",
                         delta=f"{results['speech_duration']:.1f}s")
            with col2:
                st.metric("Total Duration", f"{results['total_duration']:.2f}s")
            with col3:
                st.metric("Speech Duration", f"{results['speech_duration']:.2f}s",
                         delta=f"{results['speech_ratio']:.1%}")
            with col4:
                st.metric("Silence Duration", f"{results['silence_duration']:.2f}s")
            
            # Display segments
            st.subheader("🔊 Detected Segments")
            
            if results['segments']:
                # Create a dataframe for better display
                segments_data = []
                for i, seg in enumerate(results['segments']):
                    segments_data.append({
                        '#': i+1,
                        'Type': '🎤 SPEECH' if seg['label'] == 'speech' else '🔇 SILENCE',
                        'Start (s)': f"{seg['start']:.3f}",
                        'End (s)': f"{seg['end']:.3f}",
                        'Duration (s)': f"{seg['end']-seg['start']:.3f}",
                        'Confidence': f"{seg['confidence']:.1%}"
                    })
                
                # Display as table
                df = pd.DataFrame(segments_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Display segment count
                speech_count = sum(1 for s in results['segments'] if s['label'] == 'speech')
                silence_count = sum(1 for s in results['segments'] if s['label'] == 'silence')
                st.caption(f"Found {speech_count} speech segments and {silence_count} silence segments")
                
                # Display detailed segments in expander
                with st.expander("📝 Detailed Segment View"):
                    for i, seg in enumerate(results['segments']):
                        color = "🟢" if seg['label'] == 'speech' else "🔴"
                        st.write(f"{color} **{seg['label'].upper()}** | "
                                f"Time: `{seg['start']:.3f}s - {seg['end']:.3f}s` | "
                                f"Duration: `{seg['end']-seg['start']:.3f}s` | "
                                f"Confidence: `{seg['confidence']:.1%}`")
            else:
                st.warning("⚠️ No segments detected with current settings.")
                st.info("Try: 1) Lowering the threshold 2) Reducing minimum duration 3) Checking audio content")
            
            # Visualization
            st.subheader("📈 Visualizations")
            
            # Create tabs for different visualizations
            tab1, tab2, tab3 = st.tabs(["Waveform with Segments", "Probability Plot", "Segment Timeline"])
            
            with tab1:
                # Waveform with segments
                fig1, ax1 = plt.subplots(figsize=(12, 4))
                times_full = np.linspace(0, len(results['audio_data'])/results['sample_rate'], 
                                       len(results['audio_data']))
                ax1.plot(times_full, results['audio_data'], alpha=0.7, 
                        color='blue', linewidth=0.5, label='Audio Waveform')
                
                # Add segments
                for seg in results['segments']:
                    color = 'green' if seg['label'] == 'speech' else 'red'
                    ax1.axvspan(seg['start'], seg['end'], color=color, alpha=0.3)
                
                # Custom legend
                speech_patch = Patch(facecolor='green', alpha=0.3, label='Speech')
                silence_patch = Patch(facecolor='red', alpha=0.3, label='Silence')
                ax1.legend(handles=[speech_patch, silence_patch], loc='upper right')
                
                ax1.set_title("Audio Waveform with VAD Segments")
                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("Amplitude")
                ax1.grid(True, alpha=0.3)
                st.pyplot(fig1)
            
            with tab2:
                # Probability plot
                fig2, ax2 = plt.subplots(figsize=(12, 4))
                
                # Plot probability curve
                ax2.plot(results['times'], results['probabilities'], 'b-', 
                        alpha=0.7, linewidth=1, label='Speech Probability')
                
                # Add threshold line
                ax2.axhline(y=threshold, color='r', linestyle='--', 
                          alpha=0.7, linewidth=1.5, label=f'Threshold ({threshold})')
                
                # Fill area under curve
                ax2.fill_between(results['times'], 0, results['probabilities'], 
                               alpha=0.2, color='blue')
                
                # Highlight segments
                for seg in results['segments']:
                    if seg['label'] == 'speech':
                        ax2.axvspan(seg['start'], seg['end'], color='green', alpha=0.2)
                    else:
                        ax2.axvspan(seg['start'], seg['end'], color='red', alpha=0.2)
                
                ax2.set_title("Speech Probability over Time")
                ax2.set_xlabel("Time (s)")
                ax2.set_ylabel("Probability")
                ax2.set_ylim([-0.05, 1.05])
                ax2.legend(loc='upper right')
                ax2.grid(True, alpha=0.3)
                st.pyplot(fig2)
            
            with tab3:
                # Segment timeline visualization
                fig3, ax3 = plt.subplots(figsize=(12, 3))
                
                y_pos = 0
                for seg in results['segments']:
                    color = 'green' if seg['label'] == 'speech' else 'red'
                    ax3.barh(y_pos, seg['end']-seg['start'], left=seg['start'], 
                            height=0.6, color=color, alpha=0.7, edgecolor='black', 
                            linewidth=0.5)
                
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
                'audio_file': uploaded_file.name,
                'settings': {
                    'threshold': threshold,
                    'min_duration': min_duration,
                    'sample_rate': results['sample_rate']
                },
                'statistics': {
                    'speech_ratio': float(results['speech_ratio']),
                    'total_duration': float(results['total_duration']),
                    'speech_duration': float(results['speech_duration']),
                    'silence_duration': float(results['silence_duration']),
                    'num_segments': len(results['segments']),
                    'num_speech_segments': sum(1 for s in results['segments'] if s['label'] == 'speech'),
                    'num_silence_segments': sum(1 for s in results['segments'] if s['label'] == 'silence')
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
                    file_name=f"vad_results_{uploaded_file.name.split('.')[0]}.json",
                    mime="application/json",
                    help="Download analysis results in JSON format"
                )
            
            with col2:
                # Create CSV
                if results['segments']:
                    df_export = pd.DataFrame(results['segments'])
                    csv = df_export.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv,
                        file_name=f"vad_results_{uploaded_file.name.split('.')[0]}.csv",
                        mime="text/csv",
                        help="Download segment data in CSV format"
                    )
            
            # Debug information
            with st.expander("🔍 Debug Information"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Model Configuration:**")
                    st.write(f"- win_ctx: {analyzer.win_ctx}")
                    st.write(f"- n_feats: {analyzer.n_feats}")
                    st.write(f"- win_total: {analyzer.win_total}")
                    st.write(f"- Flat dimension: {analyzer.win_total * analyzer.n_feats}")
                
                with col2:
                    st.write("**Audio Information:**")
                    st.write(f"- Duration: {len(results['audio_data'])/results['sample_rate']:.2f}s")
                    st.write(f"- Sample rate: {results['sample_rate']} Hz")
                    st.write(f"- Samples: {len(results['audio_data']):,}")
                    st.write(f"- Probability frames: {len(results['probabilities'])}")
                
                st.write("**Probability Statistics:**")
                col3, col4 = st.columns(2)
                with col3:
                    st.write(f"- Min: {results['probabilities'].min():.3f}")
                    st.write(f"- Max: {results['probabilities'].max():.3f}")
                with col4:
                    st.write(f"- Mean: {results['probabilities'].mean():.3f}")
                    st.write(f"- Std: {results['probabilities'].std():.3f}")
                
                if st.button("Show first 20 probabilities"):
                    st.write(results['probabilities'][:20])
            
        except FileNotFoundError as e:
            st.error(f"Model files not found: {str(e)}")
            st.error("""
            Please ensure you have the following files in 'hybrid_model/' directory:
            1. hybrid_config.json
            2. scaler_ctx.joblib  
            3. bdnn.pth
            4. cnn.pth
            5. meta.pth
            """)
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            import traceback
            with st.expander("Technical Details"):
                st.code(traceback.format_exc())
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass
else:
    # Show instructions when no file is uploaded
    st.info("👆 **Upload an audio file to begin analysis**")
    
    with st.expander("🎯 How to Use"):
        st.markdown("""
        ### **Quick Start Guide:**
        
        1. **Upload Audio**
           - Click "Browse files" or drag & drop
           - Supported formats: WAV, MP3, M4A, FLAC
        
        2. **Adjust Settings** (Optional)
           - **Detection Threshold**: Controls sensitivity (0.1-0.9)
             - Lower = more speech detected (but may include noise)
             - Higher = stricter speech detection
           - **Minimum Duration**: Ignore short segments (<0.05-0.5s)
        
        3. **View Results**
           - **Statistics**: Speech ratio, durations
           - **Segment Table**: All detected segments
           - **Visualizations**: Waveform, probability plots
        
        4. **Export Data**
           - Download as JSON (full analysis)
           - Download as CSV (segment data only)
        
        ### **Tips for Best Results:**
        - Use clear speech recordings
        - Start with default settings (threshold=0.5, min_duration=0.1s)
        - Adjust threshold based on audio quality
        - For noisy audio, increase threshold slightly
        """)

st.markdown("---")
st.markdown("### 🚀 Powered by Hybrid VAD Model | Built with Streamlit")
st.caption("Voice Activity Detection using BDNN + CNN + Meta Classifier | For academic/research use")