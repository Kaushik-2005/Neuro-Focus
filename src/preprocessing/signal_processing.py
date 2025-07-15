import numpy as np
from scipy import signal
from scipy import stats
from scipy.fft import fft
from typing import Dict, List, Tuple, Optional


class EEGPreprocessor:
    """
    Handles all EEG signal preprocessing steps including filtering,
    feature extraction, and segmentation.
    """
    def __init__(self, sampling_rate: int = 128,
                 window_size: float = 1.0,
                 overlap: float = 0.5,
                 filter_ranges: Optional[Dict[str, List[float]]] = None):
        """
        Initialize preprocessor with signal parameters.
        
        Args:
            sampling_rate: Signal sampling frequency in Hz
            window_size: Analysis window size in seconds
            overlap: Overlap between consecutive windows (0-1)
            filter_ranges: Frequency ranges for different brain waves
        """
        self.sampling_rate = sampling_rate
        # Convert window size from seconds to samples
        self.window_size = max(int(window_size * sampling_rate), 32)
        # Calculate overlap in samples
        self.overlap = int(overlap * self.window_size)
        
        # Define default frequency ranges if not provided
        self.filter_ranges = filter_ranges or {
            'theta': [4, 8],    # Drowsiness, deep meditation
            'alpha': [8, 13],   # Relaxed wakefulness, closed eyes
            'beta': [13, 30]    # Active thinking, focus, alertness
        }
        
        # Create bandpass filters for each frequency range
        self.filters = self._create_filters()

    def _create_filters(self) -> Dict[str, Tuple]:
        """
        Create bandpass filters for each brain wave frequency band.
        Uses Butterworth filters for optimal frequency response.
        
        Returns:
            Dictionary of filter coefficients for each band
        """
        filters = {}
        # Nyquist frequency is half the sampling rate
        nyquist = self.sampling_rate / 2
        
        # Create filter for each frequency band
        for band, (low, high) in self.filter_ranges.items():
            # Create Butterworth bandpass filter
            # Order 2 for balance between sharpness and stability
            b, a = signal.butter(2, [low/nyquist, high/nyquist], btype='band')
            filters[band] = (b, a)
            
        return filters

    def apply_bandpass(self, data: np.ndarray, band: str) -> np.ndarray:
        """
        Apply bandpass filter to isolate specific brain wave frequencies.
        Uses zero-phase filtering to prevent time shifts.
        
        Args:
            data: Raw EEG signal
            band: Frequency band to extract ('theta', 'alpha', or 'beta')
            
        Returns:
            Filtered signal
        """
        if band not in self.filters:
            raise ValueError(f"Unknown frequency band: {band}")
            
        b, a = self.filters[band]
        
        # Add padding to reduce edge effects
        pad_size = min(len(data) // 4, 100)
        padded_data = np.pad(data, (pad_size, pad_size), mode='edge')
        
        # Apply filter forward and backward (zero phase)
        filtered = signal.filtfilt(b, a, padded_data)
        
        # Remove padding
        return filtered[pad_size:-pad_size]

    def extract_power_spectral_density(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate power spectral density using Welch's method.
        Shows how power is distributed across frequencies.
        
        Args:
            data: Input signal
            
        Returns:
            Power spectral density estimate
        """
        # Use shorter segments for better frequency resolution
        nperseg = min(len(data), self.window_size)
        if nperseg < 32:  # Ensure minimum segment length
            nperseg = 32
            
        # Calculate power spectral density
        f, psd = signal.welch(data, fs=self.sampling_rate, nperseg=nperseg)
        return psd

    def extract_time_domain_features(self, data: np.ndarray) -> Dict[str, float]:
        """
        Extract statistical features from the time domain signal.
        
        Returns dictionary of features:
            - mean: Average signal value
            - std: Signal variation
            - rms: Root mean square (signal strength)
            - kurtosis: Peakedness of distribution
            - skewness: Asymmetry of distribution
        """
        return {
            'mean': np.mean(data),  # Central tendency
            'std': np.std(data),    # Signal variation
            'rms': np.sqrt(np.mean(np.square(data))),  # Signal strength
            'kurtosis': stats.kurtosis(data),  # Peak sharpness
            'skewness': stats.skew(data)  # Distribution asymmetry
        }

    def extract_frequency_domain_features(self, data: np.ndarray) -> Dict[str, float]:
        """
        Extract features from the frequency domain.
        Calculates power in different frequency bands and ratios.
        
        Args:
            data: Input signal
            
        Returns:
            Dictionary of frequency domain features
        """
        # Calculate FFT
        fft_vals = fft(data)
        # Get frequency bins
        fft_freqs = np.fft.fftfreq(len(data), 1/self.sampling_rate)
        
        # Calculate power in each frequency band
        features = {}
        for band, (low, high) in self.filter_ranges.items():
            # Create frequency mask for this band
            mask = (fft_freqs >= low) & (fft_freqs <= high)
            # Calculate power in band
            power = np.sum(np.abs(fft_vals[mask])**2)
            features[f'{band}_power'] = power
            
        # Calculate beta/alpha ratio (attention indicator)
        # Higher ratio indicates more attention/focus
        if 'beta_power' in features and 'alpha_power' in features:
            features['beta_alpha_ratio'] = features['beta_power'] / features['alpha_power']
            
        return features

    def segment_signal(self, data: np.ndarray) -> np.ndarray:
        """
        Split signal into overlapping windows for analysis.
        Overlapping helps capture transitions between states.
        
        Args:
            data: Continuous EEG signal
            
        Returns:
            Array of signal segments
        """
        segments = []
        # Create overlapping windows
        for i in range(0, len(data) - self.window_size + 1, 
                      self.window_size - self.overlap):
            segment = data[i:i + self.window_size]
            segments.append(segment)
        return np.array(segments)

    def preprocess(self, data: np.ndarray, extract_features: bool = True) -> Dict:
        """
        Complete preprocessing pipeline.
        
        1. Ensures minimum data length
        2. Applies filters for each frequency band
        3. Extracts features if requested
        
        Args:
            data: Raw EEG signal
            extract_features: Whether to calculate features
            
        Returns:
            Dictionary containing filtered signals and features
        """
        # Ensure minimum data length
        if len(data) < 32:
            # Pad short signals by repeating edge values
            pad_size = 32 - len(data)
            data = np.pad(data, (0, pad_size), mode='edge')
            
        result = {'filtered_data': {}}
        
        # Apply filters for each frequency band
        for band in self.filter_ranges.keys():
            result['filtered_data'][band] = self.apply_bandpass(data, band)
            
        if extract_features:
            # Extract features for each frequency band
            result['features'] = {}
            for band, filtered_data in result['filtered_data'].items():
                # Calculate time and frequency domain features
                time_features = self.extract_time_domain_features(filtered_data)
                freq_features = self.extract_frequency_domain_features(filtered_data)
                
                # Store features
                result['features'][band] = {
                    'time_domain': time_features,
                    'frequency_domain': freq_features
                }
                
        return result 