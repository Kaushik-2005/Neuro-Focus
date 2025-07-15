# NeuroFocus: EEG-based Attention Detection System

## Overview
NeuroFocus is an advanced attention detection system that combines EEG (electroencephalogram) data analysis with eye state detection to accurately assess a person's attention levels. The system uses multiple brain wave patterns and sophisticated signal processing to determine cognitive engagement levels.

## Key Features
- Multi-modal EEG signal processing
- Real-time attention state classification
- Support for multiple data sources (UCI EEG, SEED, OpenBCI)
- Advanced synthetic data generation for testing
- Comprehensive feature extraction pipeline
- Multiple neural network architectures

## Technical Architecture

### 1. Data Processing Pipeline
#### Signal Preprocessing (`src/preprocessing/signal_processing.py`)
- Bandpass filtering for specific brain wave frequencies:
  - Theta (4-8 Hz): Drowsiness indicators
  - Alpha (8-13 Hz): Relaxed wakefulness
  - Beta (13-30 Hz): Active thinking and focus
- Noise reduction and artifact removal
- Dynamic window segmentation with overlap

#### Feature Extraction
- **Time Domain Features**
  - Signal mean and standard deviation
  - Root Mean Square (RMS) value
  - Kurtosis and skewness measurements
  
- **Frequency Domain Features**
  - Power Spectral Density analysis
  - Frequency band power extraction
  - Beta/Alpha ratio calculation for attention assessment

### 2. Data Sources (`src/data/`)
#### Real Data Support
- **UCI EEG Eye State Dataset**
  - 14 EEG channels
  - Binary classification (0: Eye-closed, 1: Eye-open)
  - CSV format support
  
- **SEED Dataset**
  - Emotion recognition capabilities
  - Multiple EEG channels
  - MATLAB format support
  
- **OpenBCI Integration**
  - Real-time data acquisition
  - Support for TXT and BDF formats
  - EDF/EDF+ file format support

#### Synthetic Data Generation (`src/data/synthetic_eeg.py`)
- Configurable EEG simulation
- Realistic brain wave patterns
- Artifact generation:
  - Eye blinks
  - Muscle movements
- Spatial correlation between channels
- State transitions simulation
- Attention state modeling:
  - Focused state: Enhanced beta activity
  - Unfocused state: Dominant alpha activity

### 3. Neural Network Models (`src/models/`)
#### EEGNet
- Specialized CNN architecture for EEG processing
- Features:
  - Temporal convolution layer
  - Spatial convolution layer
  - Separable convolution for efficient processing
  - Adaptive pooling for variable input sizes

#### SimpleCNN
- Lightweight architecture for real-time processing
- Components:
  - 2 convolutional layers
  - Batch normalization
  - Adaptive max pooling
  - Dropout for regularization

#### EEGMLP
- Configurable multi-layer perceptron
- Flexible architecture for feature-based classification
- Dropout layers for preventing overfitting

### 4. Attention Assessment Methodology
#### Primary Indicators
1. **Eye State Analysis**
   - Binary state detection (open/closed)
   - Correlation with attention levels

2. **Brain Wave Analysis**
   - Beta wave activity (13-30 Hz)
     - High activity indicates focused attention
     - Used as primary attention marker
   - Alpha wave activity (8-13 Hz)
     - Inverse correlation with attention
     - Relaxation state indicator
   - Beta/Alpha ratio
     - Key metric for attention assessment
     - Higher ratio indicates better focus

#### Advanced Analysis
- Feature fusion from multiple domains
- Temporal pattern recognition
- Spatial correlation analysis
- State transition detection

## Setup and Installation

### Prerequisites
- Python 3.8+
- PyTorch
- NumPy
- SciPy
- MNE
- pyEDFlib

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/NeuroFocus.git
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Preparation
```python
from src.data.data_loader import EEGDataLoader

# Initialize data loader
loader = EEGDataLoader("data_directory")

# Load UCI EEG dataset
X, y = loader.load_uci_eeg()

# Or load SEED dataset
X, y = loader.load_seed()

# For OpenBCI data
data = loader.load_openbci("file_path.txt")
```

### Training
```python
python src/models/train.py --config configs/config.yaml
```

### Real-time Inference
```python
python src/models/predict.py --model models/trained_model.pth
```

## Configuration
The system is configured through `configs/config.yaml`:
```yaml
data:
  dataset: "uci_eeg"
  sampling_rate: 128
  window_size: 1.0
  overlap: 0.5

model:
  type: "eegnet"  # Options: eegnet, cnn, mlp
  input_size: 14
  batch_size: 32
  learning_rate: 0.001
  num_epochs: 100
```

## Results and Performance
- Classification accuracy: ~85-90%
- Real-time processing capability
- Robust attention detection across subjects
- Effective artifact handling

## Future Improvements
1. Integration with additional biosignals
2. Deep learning model optimization
3. Enhanced artifact removal
4. Mobile device compatibility
5. Real-time visualization improvements

## Contributing
Contributions are welcome! Please read the contributing guidelines before submitting pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- UCI Machine Learning Repository for the EEG Eye State Dataset
- EEGNet paper authors
- OpenBCI community
- MNE-Python developers

## Contact
For questions and support, please open an issue in the GitHub repository. 