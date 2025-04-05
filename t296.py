#!/usr/bin/env python3
"""
Cryptographic Algorithm Identifier - All-in-One Script

This script combines data generation, training, and prediction in a single flow.
"""

import os
import sys
import json
import random
import logging
import numpy as np
import pandas as pd
from collections import Counter
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('crypto_identifier')

# Constants
SAMPLE_COUNT = 10  # Samples per algorithm
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

SUPPORTED_ALGORITHMS = [
    "AES-CBC-128", "AES-CBC-256",
    "AES-ECB-128", "AES-ECB-256",
    "RSA-2048", "ChaCha20"
]

#######################
# DATA GENERATION PART
#######################

def generate_key(size):
    """Generate random key"""
    return os.urandom(size // 8)

def generate_random_data(size):
    """Generate random data"""
    return os.urandom(size)

def generate_aes_sample(mode, key_size):
    """Generate AES sample"""
    try:
        from Cryptodome.Cipher import AES
        from Cryptodome.Util.Padding import pad
    except ImportError:
        from Crypto.Cipher import AES
        from Crypto.Util.Padding import pad
    
    data = generate_random_data(random.randint(64, 512))
    key = generate_key(key_size)
    
    if mode == 'ECB':
        cipher = AES.new(key, AES.MODE_ECB)
        padded_data = pad(data, AES.block_size)
        ciphertext = cipher.encrypt(padded_data)
        return {
            "algorithm": f"AES-ECB-{key_size}",
            "key_size": key_size,
            "mode": "ECB",
            "ciphertext": ciphertext.hex()
        }
    
    elif mode == 'CBC':
        iv = generate_random_data(16)
        cipher = AES.new(key, AES.MODE_CBC, iv=iv)
        padded_data = pad(data, AES.block_size)
        ciphertext = cipher.encrypt(padded_data)
        return {
            "algorithm": f"AES-CBC-{key_size}",
            "key_size": key_size,
            "mode": "CBC",
            "iv": iv.hex(),
            "ciphertext": ciphertext.hex()
        }
    
    else:
        raise ValueError(f"Unsupported AES mode: {mode}")

def generate_rsa_sample(key_size=2048):
    """Generate RSA sample"""
    try:
        from Cryptodome.PublicKey import RSA
        from Cryptodome.Cipher import PKCS1_OAEP
    except ImportError:
        from Crypto.PublicKey import RSA
        from Crypto.Cipher import PKCS1_OAEP
    
    key = RSA.generate(key_size)
    public_key = key.publickey()
    data = generate_random_data(key_size // 16)
    cipher = PKCS1_OAEP.new(public_key)
    ciphertext = cipher.encrypt(data)
    
    return {
        "algorithm": f"RSA-{key_size}",
        "key_size": key_size,
        "mode": "PKCS1",
        "ciphertext": ciphertext.hex()
    }

def generate_chacha20_sample():
    """Generate ChaCha20 sample"""
    try:
        try:
            from Cryptodome.Cipher import ChaCha20
        except ImportError:
            from Crypto.Cipher import ChaCha20
        
        data = generate_random_data(random.randint(64, 512))
        key = generate_key(256)
        nonce = generate_random_data(12)
        cipher = ChaCha20.new(key=key, nonce=nonce)
        ciphertext = cipher.encrypt(data)
        
        return {
            "algorithm": "ChaCha20",
            "key_size": 256,
            "mode": "ChaCha20",
            "nonce": nonce.hex(),
            "ciphertext": ciphertext.hex()
        }
    except (ImportError, AttributeError):
        # Fall back to AES
        return generate_aes_sample("CBC", 256)

def generate_samples(count_per_algorithm=10):
    """Generate samples for all supported algorithms"""
    samples = []
    
    for algo in SUPPORTED_ALGORITHMS:
        print(f"Generating {count_per_algorithm} samples for {algo}")
        
        for _ in range(count_per_algorithm):
            if algo.startswith("AES-"):
                parts = algo.split("-")
                mode = parts[1]
                key_size = int(parts[2])
                sample = generate_aes_sample(mode, key_size)
            elif algo.startswith("RSA-"):
                key_size = int(algo.split("-")[1])
                sample = generate_rsa_sample(key_size)
            elif algo == "ChaCha20":
                sample = generate_chacha20_sample()
            else:
                logger.warning(f"Skipping unsupported algorithm: {algo}")
                continue
            
            samples.append(sample)
    
    # Save samples
    with open(os.path.join(DATA_DIR, "training_samples.json"), "w") as f:
        json.dump(samples, f, indent=2)
    
    print(f"Generated {len(samples)} samples")
    return samples

##########################
# FEATURE EXTRACTION PART
##########################

def ensure_bytes(data):
    """Convert data to bytes"""
    if isinstance(data, str):
        try:
            return bytes.fromhex(data)
        except ValueError:
            return data.encode('utf-8')
    return data

def extract_byte_frequency(data):
    """Extract byte frequency features"""
    data_bytes = ensure_bytes(data)
    if not data_bytes:
        return {}
    
    # Count frequencies
    byte_counts = Counter(data_bytes)
    total_bytes = len(data_bytes)
    
    # Create histogram
    bins = 16
    histogram = [0] * bins
    for byte_val, count in byte_counts.items():
        bin_idx = byte_val // (256 // bins)
        histogram[bin_idx] += count / total_bytes
    
    # Create features
    features = {}
    for i, freq in enumerate(histogram):
        features[f'byte_hist_bin_{i}'] = freq
    
    # Calculate entropy
    entropy = 0.0
    for count in byte_counts.values():
        prob = count / total_bytes
        entropy -= prob * np.log2(prob)
    
    features['byte_entropy'] = entropy
    features['byte_count'] = total_bytes
    features['byte_unique_ratio'] = len(byte_counts) / 256
    
    return features

def extract_statistical_features(data):
    """Extract statistical features"""
    data_bytes = ensure_bytes(data)
    if not data_bytes:
        return {}
    
    # Convert to numerical array
    byte_values = np.array(list(data_bytes), dtype=np.uint8)
    
    # Calculate basic statistics
    features = {
        'stat_mean': float(np.mean(byte_values)),
        'stat_median': float(np.median(byte_values)),
        'stat_std': float(np.std(byte_values)),
        'stat_var': float(np.var(byte_values)),
        'stat_min': float(np.min(byte_values)),
        'stat_max': float(np.max(byte_values)),
        'stat_range': float(np.max(byte_values) - np.min(byte_values)),
    }
    
    # Calculate percentiles
    for p in [25, 50, 75]:
        features[f'stat_percentile_{p}'] = float(np.percentile(byte_values, p))
    
    return features

def extract_metadata_features(metadata):
    """Extract features from metadata"""
    features = {}
    
    # Key size
    if 'key_size' in metadata:
        key_size = int(metadata['key_size'])
        features['meta_key_size'] = key_size
        for size in [64, 128, 192, 256, 1024, 2048, 4096]:
            features[f'meta_key_size_{size}'] = 1.0 if key_size == size else 0.0
    
    # Mode
    if 'mode' in metadata:
        mode = metadata['mode']
        for m in ['ECB', 'CBC', 'CTR', 'GCM', 'CFB', 'OFB', 'PKCS1', 'ChaCha20']:
            features[f'meta_mode_{m}'] = 1.0 if mode == m else 0.0
    
    # IV/nonce presence
    features['meta_has_iv'] = 1.0 if 'iv' in metadata else 0.0
    features['meta_has_nonce'] = 1.0 if 'nonce' in metadata else 0.0
    
    return features

def extract_features(sample):
    """Extract all features from a sample"""
    features = {}
    
    # Extract from ciphertext
    ciphertext = sample.get('ciphertext', '')
    
    # Byte frequency features
    features.update(extract_byte_frequency(ciphertext))
    
    # Statistical features
    features.update(extract_statistical_features(ciphertext))
    
    # Metadata features
    metadata = {k: v for k, v in sample.items() if k != 'ciphertext'}
    features.update(extract_metadata_features(metadata))
    
    # Add algorithm label if available
    if 'algorithm' in sample:
        features['algorithm'] = sample['algorithm']
    
    return features

def extract_features_batch(samples):
    """Extract features from multiple samples"""
    print("Extracting features from samples...")
    features_list = []
    
    for i, sample in enumerate(samples):
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{len(samples)} samples")
        
        features = extract_features(sample)
        features_list.append(features)
    
    # Convert to DataFrame
    df = pd.DataFrame(features_list)
    print(f"Extracted {df.shape[1]} features from {df.shape[0]} samples")
    
    return df

###################
# MODEL TRAINING
###################

def train_model(features_df):
    """Train a model on extracted features"""
    print("Training model...")
    
    # Separate features and target
    X = features_df.drop('algorithm', axis=1)
    y = features_df['algorithm']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    accuracy = model.score(X_test_scaled, y_test)
    print(f"Model accuracy: {accuracy:.4f}")
    
    # Save model and scaler
    joblib.dump(model, os.path.join(MODELS_DIR, "model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    
    # Save feature names
    feature_names = X.columns.tolist()
    with open(os.path.join(MODELS_DIR, "feature_names.txt"), "w") as f:
        f.write("\n".join(feature_names))
    
    print(f"Model and scaler saved to {MODELS_DIR}")
    
    return model, scaler, feature_names

###################
# PREDICTION PART
###################

def load_model_and_scaler():
    """Load trained model and scaler"""
    model_path = os.path.join(MODELS_DIR, "model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return None, None
    
    model = joblib.load(model_path)
    scaler = None
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    
    return model, scaler

def find_json_files():
    """Find all JSON files in current directory"""
    return glob.glob("*.json")

def predict_from_file(file_path, model, scaler, feature_names):
    """Make predictions on samples in a JSON file"""
    print(f"\nPredicting from file: {file_path}")
    
    try:
        # Load samples
        with open(file_path, 'r') as f:
            samples = json.load(f)
        
        if not isinstance(samples, list):
            samples = [samples]
        
        print(f"Found {len(samples)} samples")
        
        # Process each sample
        for i, sample in enumerate(samples):
            ciphertext = sample.get('ciphertext', '')
            if not ciphertext:
                print(f"Sample {i+1}: Missing ciphertext, skipping")
                continue
            
            # Print basic info
            print(f"\nSample {i+1}:")
            print(f"  Ciphertext: {ciphertext[:30]}..." if len(ciphertext) > 30 else ciphertext)
            
            
            # Extract features
            features = extract_features(sample)
            
            # Align features with training data
            feature_vector = {}
            for feature in feature_names:
                feature_vector[feature] = features.get(feature, 0.0)
            
            # Convert to DataFrame with consistent column order
            df = pd.DataFrame([feature_vector])
            
            # Apply scaler if available
            if scaler is not None:
                X = scaler.transform(df)
            else:
                X = df.values
            
            # Make prediction
            prediction = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0]
            
            # Get top predictions
            top_indices = np.argsort(probabilities)[::-1][:3]
            top_classes = [model.classes_[i] for i in top_indices]
            top_probs = [probabilities[i] for i in top_indices]
            
            # Display result
            print(f"  Predicted algorithm: {prediction}")
            print(f"  Confidence: {np.max(probabilities):.4f}")
            print("  Top alternatives:")
            for algo, prob in zip(top_classes, top_probs):
                print(f"    {algo}: {prob:.4f}")
        
    except Exception as e:
        print(f"Error processing file: {e}")

###################
# MAIN FUNCTION
###################

def main():
    """Main function"""
    print("Cryptographic Algorithm Identifier - All-in-One")
    print("=" * 50)
    
    # Check if model exists
    model_path = os.path.join(MODELS_DIR, "model.pkl")
    
    # Initialize feature names
    feature_names = []
    feature_names_path = os.path.join(MODELS_DIR, "feature_names.txt")
    if os.path.exists(feature_names_path):
        with open(feature_names_path, 'r') as f:
            feature_names = [line.strip() for line in f]
    
    if os.path.exists(model_path):
        print("Existing model found!")
        model, scaler = load_model_and_scaler()
    else:
        print("No existing model found. Generating data and training new model...")
        
        # Generate samples
        samples = generate_samples(SAMPLE_COUNT)
        
        # Extract features
        features_df = extract_features_batch(samples)
        
        # Train model
        model, scaler, feature_names = train_model(features_df)
    
    # Look for JSON files to predict
    json_files = find_json_files()
    print(f"\nFound {len(json_files)} JSON files for prediction:")
    for i, file in enumerate(json_files):
        print(f"  {i+1}. {file}")
    
    # Let user select file or all
    choice = input("\nEnter file number to process, or 'all' for all files: ").strip()
    
    if choice == 'all':
        for file in json_files:
            predict_from_file(file, model, scaler, feature_names)
    elif choice.isdigit() and 1 <= int(choice) <= len(json_files):
        file = json_files[int(choice) - 1]
        predict_from_file(file, model, scaler, feature_names)
    else:
        print("Invalid choice")
    
    print("\nDone!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 