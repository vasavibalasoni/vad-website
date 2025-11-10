# app.py
import os
import io
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, render_template, jsonify, send_file
import librosa
from werkzeug.utils import secure_filename
import json
import tempfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from joblib import load
import torch.nn.functional as F

# Import your hybrid model classes
class BDNN_Head(nn.Module):
    def __init__(self, in_dim=1521, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

class CNN_VAD(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.25)
        
        # Calculate the size after convolutions and pooling
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 39, 39)
            x = F.relu(self.conv1(dummy))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = self.dropout1(x)
            conv_out = x.view(1, -1).shape[1]
            
        self.fc1 = nn.Linear(conv_out, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return self.sigmoid(x).squeeze(-1)

class MetaMLP(nn.Module):
    def __init__(self, in_dim=2, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

class HybridVADAnalyzer:
    def __init__(self, model_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir
        self.load_models()
        
    def load_models(self):
        """Load the trained hybrid model components"""
        try:
            # Load configuration
            with open(os.path.join(self.model_dir, "hybrid_config.json"), "r") as f:
                config = json.load(f)
            
            self.win_ctx = config["win_ctx"]
            self.n_feats = config["n_feats"]
            self.win_total = 2 * self.win_ctx + 1
            self.flat_dim = self.win_total * self.n_feats
            
            # Load scaler
            self.scaler = load(os.path.join(self.model_dir, "scaler_ctx.joblib"))
            
            # Initialize models
            self.bdnn = BDNN_Head(in_dim=self.flat_dim)
            self.cnn = CNN_VAD()
            self.meta = MetaMLP()
            
            # Load weights
            self.bdnn.load_state_dict(torch.load(
                os.path.join(self.model_dir, "bdnn.pth"), 
                map_location=self.device
            ))
            self.cnn.load_state_dict(torch.load(
                os.path.join(self.model_dir, "cnn.pth"), 
                map_location=self.device
            ))
            self.meta.load_state_dict(torch.load(
                os.path.join(self.model_dir, "meta.pth"), 
                map_location=self.device
            ))
            
            # Set to eval mode
            self.bdnn.to(self.device).eval()
            self.cnn.to(self.device).eval()
            self.meta.to(self.device).eval()
            
            print("✅ Hybrid VAD model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def extract_features(self, audio_path):
        """Extract MFCC features from audio file"""
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=160)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        features = np.vstack([mfcc, delta, delta2]).T
        
        return features, sr, y
    
    def add_context(self, features):
        """Add contextual windows to features"""
        padded = np.pad(features, ((self.win_ctx, self.win_ctx), (0, 0)), mode='edge')
        context_features = []
        
        for i in range(self.win_ctx, len(features) + self.win_ctx):
            window = padded[i-self.win_ctx:i+self.win_ctx+1].flatten()
            context_features.append(window)
            
        return np.array(context_features)
    
    def convert_to_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.int32):
            return int(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [self.convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self.convert_to_serializable(value) for key, value in obj.items()}
        else:
            return obj

    def probs_to_segments(self, probs, times, threshold=0.5, min_duration=0.1):
        """Exact same logic as your Colab notebook"""
        segments = []
        if len(probs) == 0:
            return segments
            
        current_label = "silence" if probs[0] < threshold else "speech"
        start_time = times[0]
        
        for i in range(1, len(probs)):
            new_label = "silence" if probs[i] < threshold else "speech"
            if new_label != current_label:
                duration = times[i] - start_time
                if duration >= min_duration:
                    # Use the same confidence calculation as Colab
                    start_frame = int(start_time * 100)
                    end_frame = int(times[i] * 100)
                    segment_probs = probs[start_frame:end_frame]
                    confidence = float(np.mean(segment_probs)) if len(segment_probs) > 0 else 0.0
                    
                    segments.append({
                        'start': float(start_time),
                        'end': float(times[i]),
                        'label': current_label,
                        'confidence': confidence
                    })
                    start_time = times[i]
                    current_label = new_label
        
        # Add final segment
        if len(probs) > 0:
            duration = times[-1] - start_time
            if duration >= min_duration:
                start_frame = int(start_time * 100)
                end_frame = int(times[-1] * 100)
                segment_probs = probs[start_frame:end_frame]
                confidence = float(np.mean(segment_probs)) if len(segment_probs) > 0 else 0.0
                
                segments.append({
                    'start': float(start_time),
                    'end': float(times[-1]),
                    'label': current_label,
                    'confidence': confidence
                })
        
        return segments
    
    def predict_vad(self, audio_path, threshold=0.5, min_duration=0.1):
        """Predict voice activity detection on audio file"""
        # Extract features
        features, sr, audio_data = self.extract_features(audio_path)
        
        # Add context and scale
        features_ctx = self.add_context(features)
        features_scaled = self.scaler.transform(features_ctx)
        
        # Prepare features for models
        features_flat = torch.tensor(features_scaled, dtype=torch.float32).to(self.device)
        features_cnn = features_flat.reshape(-1, 1, self.win_total, self.n_feats)
        
        # Get predictions
        with torch.no_grad():
            bdnn_probs = self.bdnn(features_flat).cpu().numpy()
            cnn_probs = self.cnn(features_cnn).cpu().numpy()
            
            # Meta prediction
            meta_input = torch.tensor(np.column_stack([bdnn_probs, cnn_probs]), 
                                    dtype=torch.float32).to(self.device)
            final_probs = self.meta(meta_input).cpu().numpy()
        
        # Convert to time segments
        times = librosa.frames_to_time(range(len(final_probs)), sr=sr, hop_length=160)
        segments = self.probs_to_segments(final_probs, times, threshold, min_duration)
        
        # Calculate statistics
        speech_time = sum(s['end']-s['start'] for s in segments if s['label'] == 'speech')
        total_time = segments[-1]['end'] if segments else 0
        speech_ratio = speech_time / total_time if total_time > 0 else 0
        
        results = {
            'segments': segments,
            'speech_ratio': speech_ratio,
            'total_duration': total_time,
            'speech_duration': speech_time,
            'silence_duration': total_time - speech_time,
            'frame_predictions': final_probs,
            'timestamps': times
        }
        
        # Convert all numpy types to Python native types for JSON serialization
        return self.convert_to_serializable(results)
    
    def create_visualization(self, audio_path, segments, output_path):
        """Create visualization plot"""
        y, sr = librosa.load(audio_path)
        duration = len(y) / sr
        
        plt.figure(figsize=(15, 6))
        
        # Plot waveform
        librosa.display.waveshow(y, sr=sr, alpha=0.7, color='blue')
        
        # Plot segments
        for seg in segments:
            color = 'green' if seg['label'] == 'speech' else 'red'
            plt.axvspan(seg['start'], seg['end'], color=color, alpha=0.3, 
                       label=f"{seg['label'].capitalize()}" if seg == segments[0] else "")
        
        plt.title("Voice Activity Detection Results", fontsize=16)
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("Amplitude", fontsize=12)
        plt.xlim(0, duration)
        
        # Add legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize VAD analyzer (update path to your model directory)
MODEL_DIR = "hybrid_model"  # Update this path
vad_analyzer = HybridVADAnalyzer(MODEL_DIR)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_audio():
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['audio_file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Get analysis parameters
            threshold = float(request.form.get('threshold', 0.5))
            min_duration = float(request.form.get('min_duration', 0.1))
            
            # Analyze audio
            results = vad_analyzer.predict_vad(filepath, threshold, min_duration)
            
            # Create visualization
            viz_path = os.path.join(app.config['UPLOAD_FOLDER'], 'vad_plot.png')
            vad_analyzer.create_visualization(filepath, results['segments'], viz_path)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'results': results,
                'visualization': '/plot'
            })
            
        except Exception as e:
            # Clean up uploaded file if analysis fails
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/plot')
def get_plot():
    viz_path = os.path.join(app.config['UPLOAD_FOLDER'], 'vad_plot.png')
    if os.path.exists(viz_path):
        return send_file(viz_path, mimetype='image/png')
    else:
        return "Plot not found", 404

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'wav', 'mp3', 'm4a', 'flac'}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)