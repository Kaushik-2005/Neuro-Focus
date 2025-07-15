import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import argparse
import sys

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import EEGDataLoader
from preprocessing.signal_processing import EEGPreprocessor
from models.model import get_model


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def prepare_data(config):
    """
    Prepare data for training.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Initialize data loader
    data_loader = EEGDataLoader("data")
    
    # Load or simulate data based on config
    if config['data']['dataset'] == 'uci_eeg':
        X, y = data_loader.load_uci_eeg()
    elif config['data']['dataset'] == 'seed':
        X, y = data_loader.load_seed()
    else:
        # Use simulated data for testing
        X, y = data_loader.simulate_eeg(num_samples=10000)
    
    # Preprocess data
    preprocessor = EEGPreprocessor(
        sampling_rate=config['data']['sampling_rate'],
        window_size=config['data']['window_size'],
        overlap=config['data']['overlap']
    )
    
    # Process each channel
    processed_data = []
    for i in range(X.shape[1]):
        result = preprocessor.preprocess(X[:, i])
        processed_features = []
        
        # Extract features from each frequency band
        for band in result['features']:
            time_features = list(result['features'][band]['time_domain'].values())
            freq_features = list(result['features'][band]['frequency_domain'].values())
            processed_features.extend(time_features + freq_features)
            
        processed_data.append(processed_features)
    
    # Convert to numpy arrays
    X_processed = np.array(processed_data).T
    
    # Split data
    num_samples = len(X_processed)
    train_size = int(0.7 * num_samples)
    val_size = int(0.15 * num_samples)
    
    # Shuffle indices
    indices = np.random.permutation(num_samples)
    
    # Create splits
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Convert to PyTorch tensors with proper shape
    if config['model']['type'] in ['cnn', 'eegnet']:
        # For CNN and EEGNet, reshape to [batch_size, num_channels, sequence_length]
        # Ensure minimum sequence length of 8 for the pooling operations
        min_seq_length = 8
        features_per_channel = X_processed.shape[1] // config['model']['input_size']
        seq_length = max(features_per_channel, min_seq_length)
        
        # Pad if necessary
        if features_per_channel < min_seq_length:
            pad_size = min_seq_length - features_per_channel
            X_processed = np.pad(
                X_processed,
                ((0, 0), (0, pad_size * config['model']['input_size'])),
                mode='edge'
            )
        
        # Reshape and transpose
        X_processed = X_processed.reshape(num_samples, config['model']['input_size'], -1)
        print(f"Data shape after reshape: {X_processed.shape}")
    
    X_train = torch.FloatTensor(X_processed[train_indices])
    y_train = torch.LongTensor(y[train_indices])
    
    X_val = torch.FloatTensor(X_processed[val_indices])
    y_val = torch.LongTensor(y[val_indices])
    
    X_test = torch.FloatTensor(X_processed[test_indices])
    y_test = torch.LongTensor(y[test_indices])
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=config['model']['batch_size'],
        shuffle=True
    )
    
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=config['model']['batch_size']
    )
    
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=config['model']['batch_size']
    )
    
    return train_loader, val_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in tqdm(train_loader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(train_loader)
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Validation"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(val_loader)
    
    return avg_loss, accuracy


def main(config_path):
    """Main training function."""
    # Load configuration
    config = load_config(config_path)
    
    # Set device - use CUDA if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() and config['model']['device'] == "cuda" else "cpu")
    print(f"Using device: {device}")
    
    # Prepare data
    train_loader, val_loader, test_loader = prepare_data(config)
    
    # Create model based on type
    model_type = config['model']['type']
    if model_type == 'mlp':
        model = get_model(
            model_type=model_type,
            input_size=config['model']['input_size'],
            hidden_sizes=config['model']['hidden_sizes']
        )
    else:
        # For CNN and EEGNet, use their default parameters
        model = get_model(
            model_type=model_type,
            num_channels=config['model']['input_size'],
            num_classes=2
        )
    
    model = model.to(device)
    print(f"Model architecture:\n{model}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['model']['learning_rate']
    )
    
    # Training loop
    best_val_acc = 0
    for epoch in range(config['model']['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['model']['num_epochs']}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = f"models/{config['model']['type']}_model.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model with validation accuracy: {val_acc:.2f}%")
    
    # Test best model
    model.load_state_dict(torch.load(f"models/{config['model']['type']}_model.pth"))
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EEG classification model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    args = parser.parse_args()
    
    main(args.config) 