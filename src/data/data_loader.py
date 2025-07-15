import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from scipy.io import loadmat
import mne
import pyedflib


class EEGDataLoader:
    """
    Data loader class that handles multiple EEG data formats and sources.
    Supports UCI EEG Eye State, SEED, and OpenBCI datasets.
    """
    def __init__(self, data_dir: str):
        """
        Initialize the data loader with a data directory.
        Args:
            data_dir (str): Path to directory containing dataset files
        """
        self.data_dir = data_dir  # Store the data directory path

    def load_uci_eeg(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and process the UCI EEG Eye State dataset.
        
        Returns:
            X (np.ndarray): EEG signal data of shape (n_samples, n_channels)
            y (np.ndarray): Labels (0: eyes closed, 1: eyes open)
        """
        try:
            # Construct path to the CSV file
            data_path = os.path.join(self.data_dir, "EEG_Eye_State.csv")
            
            # Check if file exists
            if not os.path.exists(data_path):
                raise FileNotFoundError("UCI EEG dataset file not found")
            
            # Read CSV file into pandas DataFrame
            df = pd.read_csv(data_path)
            
            # Split into features (X) and labels (y)
            X = df.iloc[:, :-1].values  # All columns except last
            y = df.iloc[:, -1].values   # Last column contains labels
            
            return X, y
            
        except (FileNotFoundError, pd.errors.EmptyDataError) as e:
            print(f"Warning: Could not load UCI EEG dataset ({str(e)})")
            # Generate synthetic data if real data unavailable
            return self.simulate_eeg(num_samples=1000, num_channels=14)

    def load_seed(self, subject_id: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load SEED dataset (emotion recognition dataset).
        
        Args:
            subject_id: Optional specific subject to load
            
        Returns:
            X: EEG data
            y: Emotion labels
        """
        try:
            # Get list of files to process
            if subject_id is not None:
                data_files = [f"subject_{subject_id}.mat"]
            else:
                data_files = [f for f in os.listdir(self.data_dir) if f.endswith(".mat")]
            
            if not data_files:
                raise FileNotFoundError("No SEED dataset files found")
            
            X_list, y_list = [], []
            
            # Process each subject's data file
            for file in data_files:
                data_path = os.path.join(self.data_dir, file)
                mat_data = loadmat(data_path)
                
                # Extract data and labels from MATLAB file
                X_list.append(mat_data['eeg_data'])
                y_list.append(mat_data['labels'])
            
            # Combine data from all subjects
            X = np.concatenate(X_list, axis=0)
            y = np.concatenate(y_list, axis=0)
            
            return X, y
            
        except (FileNotFoundError, KeyError) as e:
            print(f"Warning: Could not load SEED dataset ({str(e)})")
            return self.simulate_eeg(num_samples=1000, num_channels=14)

    def load_openbci(self, file_path: str) -> np.ndarray:
        """
        Load OpenBCI data from various file formats.
        
        Args:
            file_path: Path to OpenBCI data file
            
        Returns:
            np.ndarray: EEG data
        """
        # Get file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.txt':
            # Load TXT format (CSV with no header)
            data = pd.read_csv(file_path, delimiter=',', header=None)
            return data.values
            
        elif file_ext == '.bdf':
            # Load BDF format using MNE library
            raw = mne.io.read_raw_bdf(file_path, preload=True)
            return raw.get_data()
            
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

    def load_edf(self, file_path: str) -> np.ndarray:
        """
        Load EEG data from EDF/EDF+ files.
        
        Args:
            file_path: Path to EDF file
            
        Returns:
            np.ndarray: EEG data array
        """
        # Open EDF file
        f = pyedflib.EdfReader(file_path)
        
        # Get number of signals
        n = f.signals_in_file
        signal_labels = f.getSignalLabels()
        
        # Initialize array for all signals
        sigbufs = np.zeros((n, f.getNSamples()[0]))
        
        # Read each signal
        for i in range(n):
            sigbufs[i, :] = f.readSignal(i)
            
        f.close()
        return sigbufs

    def simulate_eeg(self, num_samples: int = 1000, num_channels: int = 14) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic EEG data for testing or when real data is unavailable.
        
        Args:
            num_samples: Number of time points to generate
            num_channels: Number of EEG channels to simulate
            
        Returns:
            X: Simulated EEG data
            y: Simulated labels
        """
        # Generate random EEG-like signals
        t = np.linspace(0, 10, num_samples)
        X = np.zeros((num_samples, num_channels))
        
        for i in range(num_channels):
            # Add sine waves at typical EEG frequencies
            X[:, i] = (np.sin(2 * np.pi * 10 * t) +  # Alpha wave (10 Hz)
                      0.5 * np.sin(2 * np.pi * 20 * t) +  # Beta wave (20 Hz)
                      0.3 * np.sin(2 * np.pi * 5 * t))    # Theta wave (5 Hz)
            
            # Add random noise
            X[:, i] += 0.1 * np.random.randn(num_samples)
        
        # Generate random binary labels
        y = np.random.randint(0, 2, num_samples)
        
        return X, y