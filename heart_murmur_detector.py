"""
Heart Murmur Detection Module
Based on Benjamin-Walker's Dual Bayesian ResNet implementation
Adapted for integration with the stethoscope app
"""

import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any
import joblib
import os

class BayesianResNet(nn.Module):
    """Bayesian ResNet with dropout for uncertainty estimation"""
    
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(BayesianResNet, self).__init__()
        
        # Simplified ResNet-like architecture for demo purposes
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, dropout_rate)
        self.layer2 = self._make_layer(64, 128, 2, dropout_rate, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, dropout_rate, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(256, num_classes)
        
    def _make_layer(self, in_channels, out_channels, blocks, dropout_rate, stride=1):
        layers = []
        layers.append(self._block(in_channels, out_channels, dropout_rate, stride))
        for _ in range(1, blocks):
            layers.append(self._block(out_channels, out_channels, dropout_rate))
        return nn.Sequential(*layers)
    
    def _block(self, in_channels, out_channels, dropout_rate, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout_rate)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

class HeartMurmurDetector:
    """Heart murmur detection using Dual Bayesian ResNet approach"""
    
    def __init__(self, model_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.feature_extractor = None
        self.xgb_model = None
        
        # Initialize models
        self._initialize_models()
        
        if model_path and os.path.exists(model_path):
            self.load_models(model_path)
    
    def _initialize_models(self):
        """Initialize the dual Bayesian ResNet models"""
        # Model 1: Present vs Unknown/Absent
        self.models['present_vs_other'] = BayesianResNet(num_classes=2, dropout_rate=0.5)
        
        # Model 2: Unknown vs Present/Absent  
        self.models['unknown_vs_other'] = BayesianResNet(num_classes=2, dropout_rate=0.5)
        
        # Move models to device
        for model in self.models.values():
            model.to(self.device)
            model.eval()
    
    def preprocess_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Preprocess audio for the model"""
        # Resample to 1000 Hz if needed (standard for heart sound analysis)
        if sr != 1000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=1000)
            sr = 1000
        
        # Normalize audio
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Generate mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=sr, 
            n_mels=128, 
            fmin=20, 
            fmax=1000,
            hop_length=256
        )
        
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1] range
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        
        # Add batch and channel dimensions
        mel_spec_tensor = torch.FloatTensor(mel_spec_norm).unsqueeze(0).unsqueeze(0)
        
        return mel_spec_tensor
    
    def predict_with_uncertainty(self, audio: np.ndarray, sr: int, num_samples: int = 10) -> Dict[str, Any]:
        """Predict heart murmur with uncertainty estimation using Monte Carlo dropout"""
        # Preprocess audio
        mel_spec = self.preprocess_audio(audio, sr)
        mel_spec = mel_spec.to(self.device)
        
        results = {
            'present_vs_other': {'predictions': [], 'probabilities': []},
            'unknown_vs_other': {'predictions': [], 'probabilities': []}
        }
        
        # Monte Carlo sampling for uncertainty estimation
        with torch.no_grad():
            for _ in range(num_samples):
                # Model 1: Present vs Other
                output1 = self.models['present_vs_other'](mel_spec)
                prob1 = F.softmax(output1, dim=1)
                pred1 = torch.argmax(output1, dim=1)
                
                results['present_vs_other']['predictions'].append(pred1.cpu().numpy())
                results['present_vs_other']['probabilities'].append(prob1.cpu().numpy())
                
                # Model 2: Unknown vs Other
                output2 = self.models['unknown_vs_other'](mel_spec)
                prob2 = F.softmax(output2, dim=1)
                pred2 = torch.argmax(output2, dim=1)
                
                results['unknown_vs_other']['predictions'].append(pred2.cpu().numpy())
                results['unknown_vs_other']['probabilities'].append(prob2.cpu().numpy())
        
        # Aggregate results
        final_results = self._aggregate_predictions(results)
        
        return final_results
    
    def _aggregate_predictions(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate Monte Carlo predictions and compute uncertainty"""
        aggregated = {}
        
        for model_name, model_results in results.items():
            # Stack predictions and probabilities
            preds = np.array(model_results['predictions']).squeeze()
            probs = np.array(model_results['probabilities']).squeeze()
            
            # Mean prediction and probability
            mean_pred = np.mean(preds)
            mean_prob = np.mean(probs, axis=0)
            
            # Uncertainty (standard deviation of probabilities)
            uncertainty = np.std(probs, axis=0)
            
            aggregated[model_name] = {
                'prediction': mean_pred,
                'probability': mean_prob,
                'uncertainty': uncertainty,
                'confidence': 1.0 - np.mean(uncertainty)
            }
        
        # Combine predictions for final classification
        final_class = self._combine_predictions(aggregated)
        
        return {
            'models': aggregated,
            'final_classification': final_class,
            'overall_confidence': np.mean([m['confidence'] for m in aggregated.values()])
        }
    
    def _combine_predictions(self, aggregated: Dict[str, Any]) -> Dict[str, Any]:
        """Combine predictions from both models for final classification"""
        present_prob = aggregated['present_vs_other']['probability'][1]  # Probability of present
        unknown_prob = aggregated['unknown_vs_other']['probability'][1]  # Probability of unknown
        
        # Decision logic based on the paper
        if present_prob > 0.5 and unknown_prob < 0.3:
            classification = 'Present'
            confidence = present_prob
        elif unknown_prob > 0.5:
            classification = 'Unknown'
            confidence = unknown_prob
        else:
            classification = 'Absent'
            confidence = 1.0 - max(present_prob, unknown_prob)
        
        return {
            'class': classification,
            'confidence': confidence,
            'present_probability': present_prob,
            'unknown_probability': unknown_prob
        }
    
    def load_models(self, model_path: str):
        """Load pre-trained models"""
        try:
            # Load model weights
            for model_name, model in self.models.items():
                model_file = os.path.join(model_path, f"{model_name}.pth")
                if os.path.exists(model_file):
                    model.load_state_dict(torch.load(model_file, map_location=self.device))
                    print(f"Loaded {model_name} model")
            
            # Load XGBoost model if available
            xgb_file = os.path.join(model_path, "xgboost_model.pkl")
            if os.path.exists(xgb_file):
                self.xgb_model = joblib.load(xgb_file)
                print("Loaded XGBoost model")
                
        except Exception as e:
            print(f"Warning: Could not load pre-trained models: {e}")
            print("Using randomly initialized models for demo purposes")
    
    def save_models(self, model_path: str):
        """Save trained models"""
        os.makedirs(model_path, exist_ok=True)
        
        for model_name, model in self.models.items():
            torch.save(model.state_dict(), os.path.join(model_path, f"{model_name}.pth"))
        
        if self.xgb_model:
            joblib.dump(self.xgb_model, os.path.join(model_path, "xgboost_model.pkl"))
    
    def get_detailed_analysis(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Get detailed analysis including segment-level predictions"""
        # Segment the audio
        segment_duration = 2.0  # 2-second segments
        hop_duration = 1.0      # 1-second hop
        
        segment_length = int(segment_duration * sr)
        hop_length = int(hop_duration * sr)
        
        segments = []
        segment_predictions = []
        
        for start in range(0, len(audio) - segment_length + 1, hop_length):
            end = start + segment_length
            segment = audio[start:end]
            
            # Predict for this segment
            segment_result = self.predict_with_uncertainty(segment, sr, num_samples=5)
            
            segments.append({
                'start_time': start / sr,
                'end_time': end / sr,
                'duration': segment_duration
            })
            
            segment_predictions.append(segment_result)
        
        # Overall prediction
        overall_result = self.predict_with_uncertainty(audio, sr, num_samples=20)
        
        return {
            'overall': overall_result,
            'segments': list(zip(segments, segment_predictions)),
            'audio_duration': len(audio) / sr,
            'num_segments': len(segments)
        }
