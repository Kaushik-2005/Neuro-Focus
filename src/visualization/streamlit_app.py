import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import time
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import EEGDataLoader
from data.synthetic_eeg import EEGSimulator, EEGSimulationConfig
from preprocessing.signal_processing import EEGPreprocessor
from models.model import get_model


class EEGDashboard:
    def __init__(self):
        st.set_page_config(
            page_title="NeuroFocus - Real-time Attention Monitoring",
            layout="wide"
        )
        
        # Initialize session state
        if 'eeg_data' not in st.session_state:
            st.session_state.eeg_data = []
        if 'attention_scores' not in st.session_state:
            st.session_state.attention_scores = []
        
        # Set default sampling rate
        self.sampling_rate = 128  # Default sampling rate
            
        self.setup_sidebar()
        self.initialize_components()

    def setup_sidebar(self):
        """Setup the sidebar with configuration options."""
        st.sidebar.title("NeuroFocus Settings")
        
        # Data source selection
        self.data_source = st.sidebar.selectbox(
            "Data Source",
            ["Simulation", "UCI EEG", "SEED", "OpenBCI"]
        )
        
        # Simulation settings
        if self.data_source == "Simulation":
            st.sidebar.subheader("Simulation Settings")
            
            # Basic settings
            self.sim_duration = st.sidebar.slider(
                "Duration (seconds)",
                min_value=5,
                max_value=300,
                value=60
            )
            
            self.sampling_rate = st.sidebar.slider(
                "Sampling Rate (Hz)",
                min_value=64,
                max_value=512,
                value=128
            )
            
            self.noise_level = st.sidebar.slider(
                "Noise Level",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.1
            )
            
            # Advanced settings
            st.sidebar.subheader("Advanced Simulation")
            
            self.attention_state = st.sidebar.selectbox(
                "Attention State",
                ["random", "focused", "unfocused"]
            )
            
            self.add_artifacts = st.sidebar.checkbox("Add Artifacts", value=False)
            
            if self.add_artifacts:
                self.artifact_types = st.sidebar.multiselect(
                    "Artifact Types",
                    ["blink", "muscle"],
                    default=["blink"]
                )
            
            self.add_spatial = st.sidebar.checkbox(
                "Add Spatial Correlation",
                value=False,
                help="Add realistic correlation between channels"
            )
            
            self.add_transitions = st.sidebar.checkbox(
                "Add State Transitions",
                value=False,
                help="Add smooth transitions between attention states"
            )
        else:
            # For non-simulation data sources, show sampling rate info
            if self.data_source == "UCI EEG":
                self.sampling_rate = 128  # UCI EEG dataset sampling rate
            elif self.data_source == "SEED":
                self.sampling_rate = 200  # SEED dataset sampling rate
            else:  # OpenBCI
                self.sampling_rate = 250  # Default OpenBCI sampling rate
            
            st.sidebar.info(f"Sampling Rate: {self.sampling_rate} Hz")
        
        # Model selection
        self.model_type = st.sidebar.selectbox(
            "Model Architecture",
            ["cnn", "mlp", "eegnet"]
        )
        
        # Visualization settings
        self.window_size = st.sidebar.slider(
            "Display Window (seconds)",
            min_value=5,
            max_value=60,
            value=30
        )
        
        self.update_interval = st.sidebar.slider(
            "Update Interval (seconds)",
            min_value=0.1,
            max_value=2.0,
            value=0.5,
            step=0.1
        )

    def initialize_components(self):
        """Initialize the dashboard components."""
        # Main title and model info
        st.title("NeuroFocus - Real-time Attention Monitoring")
        st.subheader(f"Model Architecture: {self.model_type.upper()}")
        
        # Create metrics at the top
        col1, col2, col3 = st.columns(3)
        with col1:
            self.attention_metric = st.empty()
        with col2:
            self.signal_quality_metric = st.empty()
        with col3:
            self.state_metric = st.empty()
        
        # Add some spacing
        st.write("")
        
        # Create placeholders for plots
        self.eeg_plot = st.empty()
        self.attention_plot = st.empty()

    def update_plots(self, eeg_data, attention_score):
        """Update the plots with new data."""
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('EEG Signal', 'Attention Score'),
            vertical_spacing=0.12,  # Reduce spacing between plots
            row_heights=[0.6, 0.4]  # Adjust relative heights
        )
        
        # Plot EEG data
        time_points = np.arange(len(eeg_data)) / self.sampling_rate
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=eeg_data,
                name='EEG',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
        
        # Plot attention scores
        attention_time = np.arange(len(st.session_state.attention_scores)) * self.update_interval
        fig.add_trace(
            go.Scatter(
                x=attention_time,
                y=st.session_state.attention_scores,
                name='Attention',
                line=dict(color='red', width=2),
                fill='tozeroy'
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=500,  # Reduce overall height
            showlegend=False,
            margin=dict(l=40, r=20, t=30, b=20),  # Adjust margins
            paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
            plot_bgcolor='rgba(0,0,0,0)',   # Transparent plot area
            font=dict(size=10)  # Slightly smaller font
        )
        
        # Update axes
        fig.update_xaxes(
            title_text='Time (s)', 
            row=1, col=1,
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=False
        )
        fig.update_xaxes(
            title_text='Time (s)', 
            row=2, col=1,
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=False
        )
        fig.update_yaxes(
            title_text='Amplitude (μV)', 
            row=1, col=1,
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=False
        )
        fig.update_yaxes(
            title_text='Attention Score', 
            range=[0, 1], 
            row=2, col=1,
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=False
        )
        
        # Display the figure
        self.eeg_plot.plotly_chart(fig, use_container_width=True)
        
        # Update metrics
        self.update_metrics(eeg_data, attention_score)

    def update_metrics(self, eeg_data, attention_score):
        """Update the dashboard metrics."""
        # Attention level metric
        attention_color = self.get_attention_color(attention_score)
        self.attention_metric.metric(
            "Attention Level",
            f"{attention_score:.2%}",
            delta=f"{(attention_score - 0.5):.1%}",
            delta_color=attention_color
        )
        
        # Signal quality metric
        signal_quality = self.calculate_signal_quality(eeg_data)
        quality_color = "normal" if signal_quality > 0.7 else "off"
        self.signal_quality_metric.metric(
            "Signal Quality",
            f"{signal_quality:.1%}",
            delta=None,
            delta_color=quality_color
        )
        
        # Attention state metric
        state = "Focused" if attention_score > 0.5 else "Unfocused"
        state_color = "normal" if attention_score > 0.5 else "off"
        self.state_metric.metric(
            "Current State",
            state,
            delta=None,
            delta_color=state_color
        )

    def get_attention_color(self, score):
        """Determine the color for attention score delta."""
        if score > 0.6:
            return "normal"  # Green
        elif score < 0.4:
            return "inverse"  # Red
        else:
            return "off"  # Gray

    def calculate_signal_quality(self, eeg_data):
        """Calculate signal quality metric."""
        # Simple signal quality based on amplitude range and variance
        amplitude_range = np.ptp(eeg_data)
        if amplitude_range < 1e-6 or amplitude_range > 1e3:
            return 0.0
        
        # Check for flatlines
        diff = np.diff(eeg_data)
        if np.any(diff == 0):
            consecutive_zeros = np.max(np.diff(np.where(diff != 0)[0]))
            if consecutive_zeros > len(eeg_data) * 0.1:  # More than 10% flatline
                return 0.3
        
        # Calculate normalized variance score
        variance = np.var(eeg_data)
        var_score = min(1.0, max(0.0, 1 - abs(np.log10(variance) + 2) / 4))
        
        # Calculate noise ratio using high-frequency content
        from scipy import signal
        freqs, psd = signal.welch(eeg_data, fs=self.sampling_rate)
        total_power = np.sum(psd)
        high_freq_power = np.sum(psd[freqs > 40])  # Power above 40 Hz
        noise_ratio = 1 - (high_freq_power / total_power if total_power > 0 else 0)
        
        # Combine metrics
        quality = (var_score + noise_ratio) / 2
        return quality

    def preprocess_data(self, eeg_data):
        """Preprocess EEG data for model input."""
        preprocessor = EEGPreprocessor(sampling_rate=self.sampling_rate)
        
        if self.model_type == 'eegnet':
            # For EEGNet: minimal preprocessing, just reshape
            # EEGNet expects shape: [batch_size, channels, time_points]
            processed_data = eeg_data.T  # [channels, time_points]
            processed_data = processed_data.reshape(1, *processed_data.shape)  # [1, channels, time_points]
            return torch.FloatTensor(processed_data)
        
        # For CNN and MLP, extract features for each channel
        processed_data = []
        for ch in range(eeg_data.shape[1]):
            channel_data = eeg_data[:, ch]
            
            # Extract features
            channel_features = []
            
            # Time domain features
            time_features = preprocessor.extract_time_domain_features(channel_data)
            channel_features.extend(list(time_features.values()))
            
            # Frequency domain features
            freq_features = preprocessor.extract_frequency_domain_features(channel_data)
            channel_features.extend(list(freq_features.values()))
            
            processed_data.append(channel_features)
        
        # Convert to numpy array
        processed_data = np.array(processed_data)
        
        if self.model_type == 'mlp':
            # For MLP: flatten all features into a single vector
            processed_data = processed_data.reshape(1, -1)
        else:  # CNN
            # For CNN: [batch_size, channels, features]
            processed_data = processed_data.reshape(1, processed_data.shape[0], -1)
        
        return torch.FloatTensor(processed_data)

    def get_simulated_data(self):
        """Generate simulated EEG data based on current settings."""
        num_samples = int(self.sim_duration * self.sampling_rate)
        
        config = EEGSimulationConfig(
            num_samples=num_samples,
            num_channels=14,
            sampling_rate=self.sampling_rate,
            add_artifacts=self.add_artifacts,
            artifact_types=self.artifact_types if self.add_artifacts else None,
            add_spatial_correlation=self.add_spatial,
            add_state_transitions=self.add_transitions,
            noise_level=self.noise_level,
            attention_state=self.attention_state
        )
        
        simulator = EEGSimulator()
        return simulator.generate_eeg(config)

    def run(self):
        """Main loop for the dashboard."""
        # Initialize data loader and model
        data_loader = EEGDataLoader("data")
        
        if self.model_type == 'mlp':
            # Calculate input size for MLP
            X_sample, _ = data_loader.simulate_eeg(num_samples=128, num_channels=14)
            features = self.preprocess_data(X_sample)
            input_size = features.shape[1]
            model = get_model(self.model_type, input_size=input_size)
        else:
            model = get_model(self.model_type)
        
        # Load model weights if available
        model_path = f"models/{self.model_type}_model.pth"
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Start button
        if st.button("Start Monitoring"):
            st.write("Monitoring started...")
            
            while True:
                # Generate or load new data
                if self.data_source == "Simulation":
                    X, _ = self.get_simulated_data()
                elif self.data_source == "UCI EEG":
                    X, _ = data_loader.load_uci_eeg()
                elif self.data_source == "SEED":
                    X, _ = data_loader.load_seed()
                else:
                    # Use simulated data for testing OpenBCI
                    X, _ = data_loader.simulate_eeg(num_samples=128, num_channels=14)
                
                # Preprocess data for model
                features = self.preprocess_data(X)
                features = features.to(device)
                
                # Get model prediction
                with torch.no_grad():
                    output = model(features)
                    attention_probs = torch.softmax(output, dim=1)
                    attention_score = attention_probs[0, 1].item()
                
                # Update session state
                st.session_state.eeg_data = X[:, 0]  # Show first channel
                st.session_state.attention_scores.append(attention_score)
                
                # Keep only last N seconds of attention scores
                max_points = int(self.window_size / self.update_interval)
                if len(st.session_state.attention_scores) > max_points:
                    st.session_state.attention_scores = st.session_state.attention_scores[-max_points:]
                
                # Update visualization
                self.update_plots(X[:, 0], attention_score)
                
                # Wait for next update
                time.sleep(self.update_interval)


if __name__ == "__main__":
    dashboard = EEGDashboard()
    dashboard.run() 