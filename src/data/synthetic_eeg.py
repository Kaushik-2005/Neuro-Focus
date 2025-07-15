import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

@dataclass
class EEGSimulationConfig:
    """Configuration for EEG simulation parameters."""
    num_samples: int = 1000
    num_channels: int = 14
    sampling_rate: int = 128
    add_artifacts: bool = False
    artifact_types: list = None
    add_spatial_correlation: bool = False
    add_state_transitions: bool = False
    noise_level: float = 0.1
    attention_state: str = 'random'  # 'focused', 'unfocused', or 'random'

class EEGSimulator:
    """Generate synthetic EEG data with customizclable characteristics."""
    
    def __init__(self):
        # Define frequency bands
        self.waves = {
            'delta': (0.5, 4),    # Deep sleep
            'theta': (4, 8),      # Drowsiness
            'alpha': (8, 13),     # Relaxed awareness
            'beta': (13, 30),     # Active thinking
            'gamma': (30, 100)    # High-level processing
        }
        
        # Define attention states
        self.states = {
            'focused': {
                'beta': 1.0,      # High beta during focus
                'alpha': 0.3,
                'theta': 0.2,
                'delta': 0.1,
                'gamma': 0.4
            },
            'unfocused': {
                'beta': 0.3,
                'alpha': 0.8,     # High alpha during relaxation
                'theta': 0.6,
                'delta': 0.4,
                'gamma': 0.1
            }
        }

    def generate_base_signal(self, config: EEGSimulationConfig) -> Tuple[np.ndarray, np.ndarray]:
        """Generate base EEG signal with realistic frequency components."""
        t = np.arange(config.num_samples) / config.sampling_rate
        X = np.zeros((config.num_samples, config.num_channels))
        y = np.zeros(config.num_samples)
        
        # Generate data in segments
        segment_length = config.sampling_rate * 2  # 2-second segments
        for i in range(0, config.num_samples, segment_length):
            end = min(i + segment_length, config.num_samples)
            segment_t = t[i:end]
            
            # Determine attention state
            if config.attention_state == 'random':
                state = np.random.choice(['focused', 'unfocused'])
            else:
                state = config.attention_state
            y[i:end] = 1 if state == 'focused' else 0
            
            for ch in range(config.num_channels):
                signal = np.zeros(len(segment_t))
                
                # Generate each frequency band
                for wave, (low_freq, high_freq) in self.waves.items():
                    num_components = 3
                    freqs = np.random.uniform(low_freq, high_freq, num_components)
                    amplitudes = np.random.uniform(0.5, 1.0, num_components)
                    phases = np.random.uniform(0, 2*np.pi, num_components)
                    
                    for f, a, p in zip(freqs, amplitudes, phases):
                        wave_component = a * np.sin(2 * np.pi * f * segment_t + p)
                        signal += wave_component * self.states[state][wave]
                
                # Add noise
                noise = np.random.normal(0, config.noise_level, len(segment_t))
                pink_noise = self._generate_pink_noise(len(segment_t))
                
                X[i:end, ch] = signal + noise + 0.1 * pink_noise
                
                # Add channel-specific characteristics
                if ch < config.num_channels // 2:  # Frontal channels
                    X[i:end, ch] *= 1.2  # Stronger frontal activity during attention
        
        return X, y

    def add_spatial_correlation(self, X: np.ndarray) -> np.ndarray:
        """Add realistic spatial correlation between EEG channels."""
        num_channels = X.shape[1]
        
        # Define correlation based on channel proximity
        correlation_matrix = np.eye(num_channels)
        for i in range(num_channels):
            for j in range(num_channels):
                if i != j:
                    distance = abs(i - j)
                    correlation_matrix[i, j] = 0.7 ** distance
        
        # Ensure correlation matrix is positive definite
        correlation_matrix = np.clip(correlation_matrix, -1, 1)
        eigenvals = np.linalg.eigvals(correlation_matrix)
        min_eigenval = np.min(eigenvals)
        if min_eigenval < 0:
            correlation_matrix += (-min_eigenval + 0.01) * np.eye(num_channels)
        
        # Apply Cholesky decomposition
        L = np.linalg.cholesky(correlation_matrix)
        
        # Apply spatial correlation
        X_corr = X @ L.T
        return X_corr

    def add_artifacts(self, X: np.ndarray, config: EEGSimulationConfig) -> np.ndarray:
        """Add realistic EEG artifacts."""
        if not config.artifact_types:
            return X
            
        X_with_artifacts = X.copy()
        
        if 'blink' in config.artifact_types:
            # Add blink artifacts
            blink_interval = config.sampling_rate * 5  # Blink every ~5 seconds
            blink_duration = config.sampling_rate // 4  # 250ms blink
            
            for i in range(0, config.num_samples, blink_interval):
                if i + blink_duration < config.num_samples:
                    blink = np.sin(np.linspace(0, np.pi, blink_duration))
                    for ch in range(config.num_channels // 3):
                        X_with_artifacts[i:i+blink_duration, ch] += blink * 2
        
        if 'muscle' in config.artifact_types:
            # Add muscle artifacts
            muscle_duration = config.sampling_rate // 2
            num_artifacts = config.num_samples // (config.sampling_rate * 10)
            
            for _ in range(num_artifacts):
                start = np.random.randint(0, config.num_samples - muscle_duration)
                artifact = np.random.normal(0, 0.5, muscle_duration)
                artifact = self._highpass_filter(artifact, config.sampling_rate)
                
                channels = np.random.choice(config.num_channels, 3, replace=False)
                for ch in channels:
                    X_with_artifacts[start:start+muscle_duration, ch] += artifact
        
        return X_with_artifacts

    def add_state_transitions(self, X: np.ndarray, y: np.ndarray, 
                            config: EEGSimulationConfig) -> Tuple[np.ndarray, np.ndarray]:
        """Add realistic transitions between attention states."""
        transition_duration = config.sampling_rate  # 1-second transition
        
        # Find state changes
        state_changes = np.where(np.diff(y) != 0)[0]
        
        for change_idx in state_changes:
            start = max(0, change_idx - transition_duration // 2)
            end = min(config.num_samples, change_idx + transition_duration // 2)
            
            # Create smooth transition in labels
            t = np.linspace(0, 1, end - start)
            y[start:end] = t if y[change_idx+1] == 1 else (1 - t)
            
            # Gradually change signal characteristics
            for ch in range(X.shape[1]):
                transition = np.linspace(X[start, ch], X[end, ch], end - start)
                X[start:end, ch] = transition
        
        return X, y

    def _generate_pink_noise(self, n_samples: int) -> np.ndarray:
        """Generate pink noise for more realistic background activity."""
        white = np.random.normal(0, 1, n_samples)
        f = np.fft.fftfreq(n_samples)
        f = np.abs(f)
        f[0] = 1e-6
        pink = np.fft.ifft(np.fft.fft(white) / np.sqrt(f))
        return pink.real

    def _highpass_filter(self, signal: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Simple high-pass filter for muscle artifacts."""
        from scipy import signal as sig
        nyquist = sampling_rate / 2
        cutoff = 20 / nyquist
        b, a = sig.butter(4, cutoff, btype='high')
        return sig.filtfilt(b, a, signal)

    def generate_eeg(self, config: EEGSimulationConfig) -> Tuple[np.ndarray, np.ndarray]:
        """Generate complete simulated EEG data based on configuration."""
        # Generate base signal
        X, y = self.generate_base_signal(config)
        
        # Add spatial correlation if requested
        if config.add_spatial_correlation:
            X = self.add_spatial_correlation(X)
        
        # Add artifacts if requested
        if config.add_artifacts:
            X = self.add_artifacts(X, config)
        
        # Add state transitions if requested
        if config.add_state_transitions:
            X, y = self.add_state_transitions(X, y, config)
        
        return X, y 