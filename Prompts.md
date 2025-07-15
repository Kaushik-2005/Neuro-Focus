# NeuroFocus Project: Prompt Used

# Project Development Prompts

## Initial Development Prompts

```
# Project Title: NeuroFocus – Real-Time Attention Detection from EEG Signals Using Deep Learning

## Problem Statement

Distractions are a major challenge in tasks like remote learning, driving, or working. Existing attention monitoring systems are either intrusive or ineffective. This project aims to build a **real-time, non-invasive EEG-based system** that classifies whether a user is **focused or distracted**, using lightweight deep learning.

---

## Objective

To develop a real-time attention classification system using **EEG signals** and **lightweight neural networks** (MLP or CNN). The model should be fast, require minimal training, and work with either simulated EEG data or consumer-grade EEG devices.

---

## Methodology

### 1. Data Collection

Use one of the following public EEG datasets:

* **EEG Eye State Dataset (UCI)**: [https://archive.ics.uci.edu/ml/datasets/EEG+Eye+State](https://archive.ics.uci.edu/ml/datasets/EEG+Eye+State)
* **SEED Dataset**: [https://bcmi.sjtu.edu.cn/\~seed/](https://bcmi.sjtu.edu.cn/~seed/)
* **OpenBCI Sample Recordings**: [https://github.com/OpenBCI/OpenBCI\_GUI/tree/master/Recordings](https://github.com/OpenBCI/OpenBCI_GUI/tree/master/Recordings)

Alternatively, simulate EEG data in CSV format for testing.

---

### 2. Preprocessing

* Apply **bandpass filters** to extract alpha, beta, theta waves.
* Use **FFT (Fast Fourier Transform)** to extract frequency-domain features.
* Segment the signal using a **sliding window** (e.g., 5 seconds).
* Normalize features per window.

---

### 3. Feature Engineering

Extract key features from each window:

* **Time domain**: Mean, standard deviation, RMS
* **Frequency domain**: Power Spectral Density (PSD) in each band (e.g., Beta/Alpha ratio)

---

### 4. Model Architecture

* Use a **Multi-Layer Perceptron (MLP)** or **shallow CNN** for binary classification:
  `0 = Distracted`, `1 = Focused`
* Optionally explore **EEGNet**, a CNN architecture optimized for EEG.

---

### 5. Real-Time Inference

* Simulate real-time EEG streaming or use a device like Muse/Emotiv/OpenBCI.
* Predict focus status every few seconds using a sliding window.
* Display results on a **Streamlit or Flask dashboard**.

---

## Reference GitHub Repositories

* 🔗 [Khalizo/Deep-Learning-Detection-Of-EEG-Based-Attention](https://github.com/Khalizo/Deep-Learning-Detection-Of-EEG-Based-Attention)
  CNN-RNN model for EEG-based attention detection with \~93% accuracy.

* 🔗 [kei5uke/focus-detection](https://github.com/kei5uke/focus-detection)
  Simple EEG-based focus detection from real-world student activities.

* 🔗 [felixmaldonadoos/focus-tracker](https://github.com/felixmaldonadoos/focus-tracker)
  Real-time EEG focus tracker with UI and data processing pipeline.

* 🔗 [D1o0g9s/MyND](https://github.com/D1o0g9s/MyND)
  Real-time EEG focus/distracted detection with OpenBCI + eye-tracking.

* 🔗 [torcheeg/TorchEEG](https://github.com/torcheeg/torcheeg)
  High-level EEG analysis framework using PyTorch.

---

## Technologies Used

* Python, NumPy, SciPy, Pandas
* TensorFlow or PyTorch
* MNE-Python, pyEDFlib (for EEG processing)
* Streamlit or Flask (for real-time dashboard)
* (Optional) EEG hardware: OpenBCI, Muse, Emotiv Insight

---

## Outcome

A lightweight, efficient, and real-time EEG-based focus detection system with applications in:

* E-learning engagement tracking
* Focus coaching and mindfulness
* Driver alertness monitoring
* Human-computer interaction and brain-computer interface (BCI) demos
```
## Error Fixing Prompts

### 1. Network Architecture Size/Shape Issues
```
I'm getting a size mismatch error in the EEGNet model:
RuntimeError: Given groups=1, weight of size [8, 1, 1, 32], expected input[64, 14, 128] to have 1 channels, but got 14 channels instead

Can you help fix the network architecture to handle the input shape correctly? The input data has 14 channels and variable sequence length.
```

### 2. Data Shape Handling
```
The SimpleCNN model is giving errors with input shape:
RuntimeError: size mismatch, got 1-dimensional input of size [14] but expected 2 dimensions

How can we modify the model to properly handle the EEG data shape? Each sample has 14 channels and the sequence length varies.
```

### 3. EEG Signal Simulation Improvement
```
The simulated EEG signals don't look realistic enough. They're too regular and don't have the natural variations seen in real EEG. Can you help improve the synthetic_eeg.py to:
1. Add more realistic noise patterns
2. Include proper state transitions
3. Add realistic artifacts like eye blinks and muscle movement
4. Make the frequency distributions more natural
```

### 4. Artifact Generation Enhancement
```
The current artifact simulation is too simplistic. Real EEG has more complex artifacts. Can you modify the add_artifacts function to:
1. Generate more realistic eye blink patterns
2. Add muscle movement artifacts that look more natural
3. Include electrode pop artifacts
4. Add proper temporal correlation to the artifacts
```

### 5. UI/UX Improvements
```
The current visualization interface is basic and hard to understand. Can you enhance the streamlit_app.py to:
1. Add real-time attention level visualization
2. Show brain wave power distributions
3. Create a more intuitive layout
4. Add interactive controls for filtering and analysis
5. Include proper error handling and user feedback
```

### 6. Model Training Issues
```
The model training seems unstable with high loss values and inconsistent accuracy. Can you help modify the training process to:
1. Add proper batch normalization
2. Implement better learning rate scheduling
3. Add early stopping
4. Handle class imbalance
5. Implement proper validation splits
```

### 7. Data Preprocessing Enhancement
```
The signal preprocessing needs improvement. Can you modify the EEGPreprocessor class to:
1. Add better artifact rejection
2. Implement proper baseline correction
3. Add more robust filtering options
4. Include signal quality checks
5. Add proper normalization methods
```

### 8. Performance Optimization
```
The system is running too slowly for real-time analysis. Can you help optimize:
1. The data loading pipeline
2. The preprocessing steps
3. The model inference
4. The visualization updates
```