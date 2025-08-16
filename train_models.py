"""
Training script for Heart Murmur Detection Models
Based on Benjamin-Walker's Dual Bayesian ResNet approach
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import librosa
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from tqdm import tqdm

from heart_murmur_detector import BayesianResNet, HeartMurmurDetector

class HeartSoundDataset(Dataset):
    """Dataset for heart sound recordings"""
    
    def __init__(self, audio_files, labels, sr=1000, segment_duration=2.0):
        self.audio_files = audio_files
        self.labels = labels
        self.sr = sr
        self.segment_duration = segment_duration
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        # Load audio file
        audio_file = self.audio_files[idx]
        label = self.labels[idx]
        
        try:
            # Load and preprocess audio
            y, sr = librosa.load(audio_file, sr=self.sr)
            
            # Normalize
            if np.max(np.abs(y)) > 0:
                y = y / np.max(np.abs(y))
            
            # Generate mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y, 
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
            
            # Resize spectrogram to fixed size (128x128)
            target_length = 128
            current_length = mel_spec_norm.shape[1]
            
            if current_length > target_length:
                # If too long, take the middle portion
                start = (current_length - target_length) // 2
                mel_spec_norm = mel_spec_norm[:, start:start + target_length]
            elif current_length < target_length:
                # If too short, pad with zeros
                padding = target_length - current_length
                mel_spec_norm = np.pad(mel_spec_norm, ((0, 0), (0, padding)), mode='constant')
            
            # Ensure exact size
            mel_spec_norm = mel_spec_norm[:, :target_length]
            
            # Convert to tensor
            mel_spec_tensor = torch.FloatTensor(mel_spec_norm).unsqueeze(0)
            
            return mel_spec_tensor, torch.LongTensor([label])
            
        except Exception as e:
            print(f"Error loading {audio_file}: {e}")
            # Return a dummy tensor
            dummy_spec = torch.zeros(1, 128, 128)
            return dummy_spec, torch.LongTensor([label])

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
    """Train a single model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_specs, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch_specs = batch_specs.to(device)
            batch_labels = batch_labels.squeeze().to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_specs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_specs, batch_labels in val_loader:
                batch_specs = batch_specs.to(device)
                batch_labels = batch_labels.squeeze().to(device)
                
                outputs = model(batch_specs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()
        
        val_acc = val_correct / val_total
        val_accuracies.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(avg_train_loss)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"New best validation accuracy: {best_val_acc:.4f}")
    
    return model, train_losses, val_accuracies

def prepare_data(data_directory, labels_file):
    """Prepare data for training from CirCor dataset"""
    print("Loading CirCor dataset...")
    
    try:
        # Load the CSV file with labels
        df = pd.read_csv(labels_file)
        print(f"Loaded {len(df)} records from labels file")
        
        # Check the structure of the CSV
        print("CSV columns:", df.columns.tolist())
        print("First few rows:")
        print(df.head())
        
        # The CirCor dataset typically has columns like:
        # - recording_name: name of the audio file
        # - murmur: murmur status (Present, Absent, Unknown)
        # - outcome: clinical outcome
        
        # Find the murmur column (it might be named differently)
        murmur_col = None
        for col in df.columns:
            if 'murmur' in col.lower():
                murmur_col = col
                break
        
        if murmur_col is None:
            print("Warning: Could not find murmur column. Available columns:")
            for col in df.columns:
                print(f"  - {col}")
            # Try to guess the column
            if 'Present' in str(df.iloc[0]):
                for col in df.columns:
                    if df[col].dtype == 'object' and 'Present' in df[col].values:
                        murmur_col = col
                        print(f"Guessing murmur column: {col}")
                        break
        
        if murmur_col is None:
            raise ValueError("Could not identify murmur column in CSV")
        
        # Convert murmur labels to numeric
        # For binary classification, we'll treat Unknown as a separate case
        label_mapping = {'Present': 1, 'Absent': 0, 'Unknown': 2}
        df['murmur_numeric'] = df[murmur_col].map(label_mapping)
        
        # Create binary labels for each model
        # Model 1: Present (1) vs Other (0)
        df['present_vs_other'] = (df['murmur_numeric'] == 1).astype(int)
        
        # Model 2: Unknown (1) vs Other (0) 
        df['unknown_vs_other'] = (df['murmur_numeric'] == 2).astype(int)
        
        # Find the recording name column
        recording_col = None
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['recording', 'file', 'name', 'id']):
                recording_col = col
                break
        
        if recording_col is None:
            # If no obvious column, use the first column that looks like filenames
            for col in df.columns:
                if df[col].dtype == 'object' and df[col].str.contains('\.').any():
                    recording_col = col
                    print(f"Guessing recording column: {col}")
                    break
        
        if recording_col is None:
            raise ValueError("Could not identify recording column in CSV")
        
        # Build file paths and labels
        audio_files = []
        labels = []
        
        # We'll create two separate datasets for the two models
        # Model 1: Present vs Other
        present_audio_files = []
        present_labels = []
        
        # Model 2: Unknown vs Other  
        unknown_audio_files = []
        unknown_labels = []
        
        for idx, row in df.iterrows():
            patient_id = str(row[recording_col])
            present_label = row['present_vs_other']
            unknown_label = row['unknown_vs_other']
            
            # The CirCor dataset has files named like: 50053_AV.wav, 50053_PV.wav, etc.
            # We need to find any .wav file that starts with the patient ID
            found_files = []
            
            # Look for files that start with the patient ID
            for filename in os.listdir(data_directory):
                if filename.startswith(patient_id + '_') and filename.endswith('.wav'):
                    found_files.append(filename)
            
            # If no files found with underscore, try exact match
            if not found_files:
                for filename in os.listdir(data_directory):
                    if filename.startswith(patient_id) and filename.endswith('.wav'):
                        found_files.append(filename)
            
            # Add all found files for this patient to both datasets
            for filename in found_files:
                audio_file = os.path.join(data_directory, filename)
                
                # Add to present vs other dataset
                present_audio_files.append(audio_file)
                present_labels.append(present_label)
                
                # Add to unknown vs other dataset
                unknown_audio_files.append(audio_file)
                unknown_labels.append(unknown_label)
            
            if not found_files:
                print(f"Warning: Could not find audio files for patient {patient_id}")
        
        print(f"Found {len(present_audio_files)} audio files for Present vs Other model")
        print(f"Present vs Other label distribution: {pd.Series(present_labels).value_counts().to_dict()}")
        print(f"Found {len(unknown_audio_files)} audio files for Unknown vs Other model")
        print(f"Unknown vs Other label distribution: {pd.Series(unknown_labels).value_counts().to_dict()}")
        
        return present_audio_files, present_labels, unknown_audio_files, unknown_labels
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please check your file paths and CSV structure")
        return [], []

def main():
    """Main training function"""
    print("Heart Murmur Detection Model Training")
    print("=" * 50)
    
    # Configuration
    data_directory = r"C:\Users\mumina or numan idc\Downloads\the-circor-digiscope-phonocardiogram-dataset-1.0.3\the-circor-digiscope-phonocardiogram-dataset-1.0.3\training_data"
    labels_file = r"C:\Users\mumina or numan idc\Downloads\the-circor-digiscope-phonocardiogram-dataset-1.0.3\the-circor-digiscope-phonocardiogram-dataset-1.0.3\training_data.csv"
    model_save_path = "trained_models"
    num_epochs = 50
    learning_rate = 0.001
    batch_size = 32
    
    # Check if data exists
    if not os.path.exists(data_directory):
        print(f"Data directory not found: {data_directory}")
        print("Please update the data_directory path in this script")
        return
    
    # Prepare data
    print("Preparing data...")
    present_files, present_labels, unknown_files, unknown_labels = prepare_data(data_directory, labels_file)
    
    if len(present_files) == 0 or len(unknown_files) == 0:
        print("No audio files found. Please check your data directory and labels file.")
        return
    
    # Split data for Present vs Other model
    present_train_files, present_temp_files, present_train_labels, present_temp_labels = train_test_split(
        present_files, present_labels, test_size=0.3, random_state=42, stratify=present_labels
    )
    
    present_val_files, present_test_files, present_val_labels, present_test_labels = train_test_split(
        present_temp_files, present_temp_labels, test_size=0.5, random_state=42, stratify=present_temp_labels
    )
    
    # Split data for Unknown vs Other model
    unknown_train_files, unknown_temp_files, unknown_train_labels, unknown_temp_labels = train_test_split(
        unknown_files, unknown_labels, test_size=0.3, random_state=42, stratify=unknown_labels
    )
    
    unknown_val_files, unknown_test_files, unknown_val_labels, unknown_test_labels = train_test_split(
        unknown_temp_files, unknown_temp_labels, test_size=0.5, random_state=42, stratify=unknown_temp_labels
    )
    
    print(f"Present vs Other - Train: {len(present_train_files)}, Validation: {len(present_val_files)}, Test: {len(present_test_files)}")
    print(f"Unknown vs Other - Train: {len(unknown_train_files)}, Validation: {len(unknown_val_files)}, Test: {len(unknown_test_files)}")
    
    # Create datasets for Present vs Other
    present_train_dataset = HeartSoundDataset(present_train_files, present_train_labels)
    present_val_dataset = HeartSoundDataset(present_val_files, present_val_labels)
    present_test_dataset = HeartSoundDataset(present_test_files, present_test_labels)
    
    # Create datasets for Unknown vs Other
    unknown_train_dataset = HeartSoundDataset(unknown_train_files, unknown_train_labels)
    unknown_val_dataset = HeartSoundDataset(unknown_val_files, unknown_val_labels)
    unknown_test_dataset = HeartSoundDataset(unknown_test_files, unknown_test_labels)
    
    # Create dataloaders for Present vs Other
    present_train_loader = DataLoader(present_train_dataset, batch_size=batch_size, shuffle=True)
    present_val_loader = DataLoader(present_val_dataset, batch_size=batch_size, shuffle=False)
    present_test_loader = DataLoader(present_test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create dataloaders for Unknown vs Other
    unknown_train_loader = DataLoader(unknown_train_dataset, batch_size=batch_size, shuffle=True)
    unknown_val_loader = DataLoader(unknown_val_dataset, batch_size=batch_size, shuffle=False)
    unknown_test_loader = DataLoader(unknown_test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize models
    print("Initializing models...")
    present_vs_other = BayesianResNet(num_classes=2, dropout_rate=0.5)
    unknown_vs_other = BayesianResNet(num_classes=2, dropout_rate=0.5)
    
    # Train models
    print("\nTraining Present vs Other model...")
    present_vs_other, train_losses1, val_accuracies1 = train_model(
        present_vs_other, present_train_loader, present_val_loader, num_epochs, learning_rate
    )
    
    print("\nTraining Unknown vs Other model...")
    unknown_vs_other, train_losses2, val_accuracies2 = train_model(
        unknown_vs_other, unknown_train_loader, unknown_val_loader, num_epochs, learning_rate
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    present_vs_other.eval()
    unknown_vs_other.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Evaluate Present vs Other model
    present_test_predictions = []
    present_test_true_labels = []
    
    with torch.no_grad():
        for batch_specs, batch_labels in present_test_loader:
            batch_specs = batch_specs.to(device)
            batch_labels = batch_labels.squeeze().to(device)
            
            outputs = present_vs_other(batch_specs)
            _, predicted = torch.max(outputs.data, 1)
            present_test_predictions.extend(predicted.cpu().numpy())
            present_test_true_labels.extend(batch_labels.cpu().numpy())
    
    # Evaluate Unknown vs Other model
    unknown_test_predictions = []
    unknown_test_true_labels = []
    
    with torch.no_grad():
        for batch_specs, batch_labels in unknown_test_loader:
            batch_specs = batch_specs.to(device)
            batch_labels = batch_labels.squeeze().to(device)
            
            outputs = unknown_vs_other(batch_specs)
            _, predicted = torch.max(outputs.data, 1)
            unknown_test_predictions.extend(predicted.cpu().numpy())
            unknown_test_true_labels.extend(batch_labels.cpu().numpy())
    
    # Calculate accuracies
    present_acc = accuracy_score(present_test_true_labels, present_test_predictions)
    unknown_acc = accuracy_score(unknown_test_true_labels, unknown_test_predictions)
    
    print(f"Test Accuracy - Present vs Other: {present_acc:.4f}")
    print(f"Test Accuracy - Unknown vs Other: {unknown_acc:.4f}")
    
    # Save models
    print(f"\nSaving models to {model_save_path}...")
    os.makedirs(model_save_path, exist_ok=True)
    
    torch.save(present_vs_other.state_dict(), os.path.join(model_save_path, "present_vs_other.pth"))
    torch.save(unknown_vs_other.state_dict(), os.path.join(model_save_path, "unknown_vs_other.pth"))
    
    print("Training completed!")
    print(f"Models saved to: {model_save_path}")
    print("\nTo use these models in your app:")
    print("1. Update the model_path in HeartMurmurDetector")
    print("2. Or copy the .pth files to your app directory")

if __name__ == "__main__":
    main()
