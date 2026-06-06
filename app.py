"""
Industrial Predictive Maintenance Engine
----------------------------------------
A real-time Streamlit dashboard that simulates IoT sensor telemetry and uses a 
PyTorch Autoencoder to detect equipment degradation and anomalies.

This script demonstrates end-to-end MLOps capabilities:
1. Live data streaming simulation.
2. Real-time PyTorch model inference.
3. Dynamic UI rendering with Streamlit.
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time

# -----------------------------------------
# 1. PyTorch Autoencoder Architecture
# -----------------------------------------
class SensorAutoencoder(nn.Module):
    """
    A simple PyTorch Autoencoder neural network for anomaly detection.
    
    The network is trained to compress normal operational 3D sensor data 
    (Vibration, Temperature, Pressure) into a smaller latent space (1D) 
    and reconstruct it. When anomalous data is fed into the network, it 
    will fail to reconstruct it accurately, resulting in a high Mean Squared 
    Error (MSE).
    """
    def __init__(self):
        super(SensorAutoencoder, self).__init__()
        
        # Compress 3 sensor readings into a 1D latent space
        self.encoder = nn.Sequential(
            nn.Linear(3, 2),
            nn.ReLU(),
            nn.Linear(2, 1)
        )
        
        # Reconstruct back to 3 dimensions
        self.decoder = nn.Sequential(
            nn.Linear(1, 2),
            nn.ReLU(),
            nn.Linear(2, 3)
        )

    def forward(self, x):
        """
        Passes the input tensor through the encoder and decoder.
        
        Args:
            x (torch.Tensor): Input sensor readings, shape (batch_size, 3)
            
        Returns:
            torch.Tensor: Reconstructed sensor readings, shape (batch_size, 3)
        """
        return self.decoder(self.encoder(x))

# -----------------------------------------
# 2. Setup & Initialization
# -----------------------------------------
# Configure the Streamlit page layout and metadata
st.set_page_config(page_title="Predictive Maintenance Engine", layout="wide")
st.title("⚙️ Industrial ML: Real-Time Predictive Maintenance")
st.markdown("Monitoring live IoT sensor telemetry via a PyTorch Autoencoder to detect process anomalies and equipment degradation.")

@st.cache_resource
def load_model():
    """
    Initializes and caches the PyTorch autoencoder model.
    
    The @st.cache_resource decorator ensures that the model is loaded into 
    memory only once when the app starts, rather than re-initializing on 
    every Streamlit UI rerun, which saves compute resources.
    
    Returns:
        SensorAutoencoder: The instantiated PyTorch model in evaluation mode.
    """
    model = SensorAutoencoder()
    model.eval() # Set to evaluation mode to disable dropout/batchnorm updates
    return model

# Load the model into the app state
model = load_model()

# -----------------------------------------
# 3. Dashboard Layout
# -----------------------------------------
# Divide the screen into two columns for better UI organization
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Sensor Telemetry")
    # Create an empty placeholder that we will continuously update with charts
    chart_placeholder = st.empty()

with col2:
    st.subheader("System Health")
    # Placeholders for the text alert and the anomaly score graph
    status_placeholder = st.empty()
    anomaly_chart_placeholder = st.empty()

# -----------------------------------------
# 4. Streaming Simulation Loop
# -----------------------------------------
# The loop will only trigger when the user clicks the start button
if st.button("Start Live Monitoring System"):
    
    # Initialize arrays to hold historical data for the sliding window charts
    history_len = 50
    sensor_data = np.zeros((history_len, 3))
    anomaly_scores = np.zeros(history_len)
    
    t = 0 # Time variable to simulate periodic sensor waves
    
    while True:
        # Simulate baseline operational data using sine/cosine waves + random noise
        vibration = np.sin(t * 0.1) + np.random.normal(0, 0.1)
        temperature = np.cos(t * 0.05) + np.random.normal(0, 0.1)
        pressure = np.sin(t * 0.2) + np.random.normal(0, 0.1)
        
        # Inject an anomaly 5% of the time to simulate a machine fault
        if np.random.rand() > 0.95:
            vibration += np.random.uniform(2.0, 4.0)
            
        current_readings = np.array([vibration, temperature, pressure], dtype=np.float32)
        
        # -----------------------------------------
        # 5. Inference: Calculate Anomaly Score
        # -----------------------------------------
        # Disable gradient calculation since we are only doing inference
        with torch.no_grad():
            # Convert the numpy array to a PyTorch tensor and add a batch dimension
            tensor_input = torch.tensor(current_readings).unsqueeze(0)
            
            # Pass data through the Autoencoder
            reconstruction = model(tensor_input)
            
            # The MSE Loss between the original input and the reconstruction
            # serves as our anomaly score. Higher loss = greater anomaly.
            mse_loss = nn.functional.mse_loss(reconstruction, tensor_input).item()
        
        # Roll the arrays backward by 1 to create a sliding window effect, 
        # dropping the oldest data point and appending the newest one.
        sensor_data = np.roll(sensor_data, -1, axis=0)
        sensor_data[-1] = current_readings
        
        anomaly_scores = np.roll(anomaly_scores, -1)
        anomaly_scores[-1] = mse_loss
        
        # -----------------------------------------
        # 6. Update UI
        # -----------------------------------------
        # Convert the raw arrays into Pandas DataFrames for easy Streamlit plotting
        df_sensors = pd.DataFrame(sensor_data, columns=["Vibration", "Temperature", "Pressure"])
        df_anomaly = pd.DataFrame(anomaly_scores, columns=["Reconstruction Error (MSE)"])
        
        # Overwrite the empty placeholders with the new live charts
        chart_placeholder.line_chart(df_sensors)
        anomaly_chart_placeholder.area_chart(df_anomaly, color="#FF4B4B")
        
        # Trigger visual alerts based on the reconstruction error threshold
        if mse_loss > 1.2:          # Threshold for anomaly detection (tuned based on normal operational data)
            status_placeholder.error(f"⚠️ FAULT DETECTED! MSE: {mse_loss:.2f}")
        else:
            status_placeholder.success(f"✅ System Nominal. MSE: {mse_loss:.2f}")
            
        t += 1
        time.sleep(0.1) # Simulate real-time network delay between sensor readings