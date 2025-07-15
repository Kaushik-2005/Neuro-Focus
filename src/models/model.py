import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    """
    EEGNet: Specialized CNN architecture for EEG signal processing.
    Based on the paper: EEGNet: A Compact CNN for EEG-based BCIs
    
    Key features:
    - Separable convolutions for efficient processing
    - Temporal and spatial filtering stages
    - Compact architecture with few parameters
    """
    def __init__(self, num_channels=14, num_classes=2):
        """
        Initialize EEGNet architecture.
        
        Args:
            num_channels: Number of EEG channels (default 14)
            num_classes: Number of output classes (default 2 for binary)
        """
        super(EEGNet, self).__init__()
        
        # Layer 1: Temporal Convolution
        # Learns frequency filters (similar to bandpass filters)
        self.conv1 = nn.Conv2d(1, 8, (1, 32), padding=(0, 16))
        self.batchnorm1 = nn.BatchNorm2d(8)
        
        # Layer 2: Spatial Convolution
        # Learns spatial filters (channel relationships)
        self.conv2 = nn.Conv2d(8, 16, (num_channels, 1), 
                              padding=(num_channels//2, 0))
        self.batchnorm2 = nn.BatchNorm2d(16)
        
        # Layer 3: Separable Convolution
        # Efficient temporal summaries
        self.conv3 = nn.Conv2d(16, 16, (1, 8), padding=(0, 4), groups=16)
        self.conv4 = nn.Conv2d(16, 16, (1, 1))  # Pointwise conv
        self.batchnorm3 = nn.BatchNorm2d(16)
        
        # Pooling layers for dimensionality reduction
        self.temporal_pool = nn.AdaptiveAvgPool2d((num_channels, 8))
        self.final_pool = nn.AdaptiveAvgPool2d((1, 4))
        
        # Classification layer
        self.fc = nn.Linear(16 * 4, num_classes)
        
        # Dropout for regularization (50% dropout rate)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch, channels, time]
            
        Returns:
            Classification logits
        """
        # Ensure 4D input [batch, 1, time, channels]
        if len(x.shape) == 3:
            x = x.permute(0, 2, 1)  # [batch, time, channels]
            x = x.unsqueeze(1)      # Add channel dimension
        
        # Block 1: Temporal Convolution
        x = self.conv1(x)           # Learn frequency filters
        x = self.batchnorm1(x)      # Normalize
        x = F.elu(x)                # Non-linearity
        x = self.temporal_pool(x)   # Reduce temporal dimension
        x = self.dropout(x)         # Prevent overfitting
        
        # Block 2: Spatial Convolution
        x = self.conv2(x)           # Learn spatial filters
        x = self.batchnorm2(x)      # Normalize
        x = F.elu(x)                # Non-linearity
        x = self.dropout(x)         # Prevent overfitting
        
        # Block 3: Separable Convolution
        x = self.conv3(x)           # Depthwise temporal conv
        x = self.conv4(x)           # Pointwise conv
        x = self.batchnorm3(x)      # Normalize
        x = F.elu(x)                # Non-linearity
        x = self.final_pool(x)      # Final dimension reduction
        x = self.dropout(x)         # Prevent overfitting
        
        # Classification
        x = x.view(x.size(0), -1)   # Flatten
        x = self.fc(x)              # Linear classification
        
        return x


class SimpleCNN(nn.Module):
    """
    Lightweight CNN for EEG classification.
    Simpler architecture for faster processing.
    
    Features:
    - 1D convolutions for temporal processing
    - Batch normalization for stable training
    - Adaptive pooling for variable input sizes
    """
    def __init__(self, num_channels=14, num_classes=2):
        super(SimpleCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv1d(num_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        
        # Second convolutional block
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        
        # Adaptive pooling (handles variable input lengths)
        self.pool = nn.AdaptiveMaxPool1d(4)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor [batch, channels, time]
            
        Returns:
            Classification logits
        """
        # First conv block
        x = self.conv1(x)           # Temporal convolution
        x = self.bn1(x)             # Normalize
        x = F.relu(x)               # Non-linearity
        
        # Second conv block
        x = self.conv2(x)           # More temporal features
        x = self.bn2(x)             # Normalize
        x = F.relu(x)               # Non-linearity
        
        # Adaptive pooling
        x = self.pool(x)            # Reduce to fixed size
        
        # Fully connected layers
        x = x.view(x.size(0), -1)   # Flatten
        x = F.relu(self.fc1(x))     # Hidden layer
        x = self.dropout(x)         # Prevent overfitting
        x = self.fc2(x)             # Classification
        
        return x


class EEGMLP(nn.Module):
    """
    Simple MLP for EEG classification.
    Most basic architecture, good baseline model.
    
    Features:
    - Flexible hidden layer sizes
    - ReLU activation
    - Dropout regularization
    """
    def __init__(self, input_size, hidden_sizes=[64, 32], num_classes=2):
        super(EEGMLP, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Create hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),  # Linear transformation
                nn.ReLU(),                          # Non-linearity
                nn.Dropout(0.5)                     # Regularization
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        # Combine all layers
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor [batch, features]
            
        Returns:
            Classification logits
        """
        # Flatten input if needed
        x = x.view(x.size(0), -1)
        return self.model(x)


def get_model(model_type="cnn", **kwargs):
    """
    Factory function to create model instances.
    
    Args:
        model_type: Type of model to create ("cnn", "mlp", or "eegnet")
        **kwargs: Additional arguments for model constructor
    
    Returns:
        Instantiated model
    
    Raises:
        ValueError: If model_type is not recognized
    """
    models = {
        "cnn": SimpleCNN,      # Fast, simple model
        "mlp": EEGMLP,         # Basic baseline
        "eegnet": EEGNet       # Advanced architecture
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return models[model_type](**kwargs) 