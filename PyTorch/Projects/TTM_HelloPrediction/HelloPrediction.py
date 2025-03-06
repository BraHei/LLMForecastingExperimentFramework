import numpy as np
from tsfm.modeling import GraniteModel
import torch

# Step 1: Load the TinyTimeMixer model
def load_model():
    print("Loading TinyTimeMixer model...")
    model = GraniteModel.from_pretrained("ibm-granite/granite-timeseries-ttm-r1")
    return model

# Step 2: Generate sine wave input
def generate_sine_wave(seq_length, num_samples):
    """Generates sine wave data for testing."""
    x = np.linspace(0, 2 * np.pi, seq_length)
    data = [np.sin(x + i) for i in range(num_samples)]
    return np.array(data)

# Step 3: Prepare data for the model
def prepare_data(data):
    """Prepares data for model inference."""
    # Normalize or scale the sine wave data as needed by the model
    data = torch.tensor(data, dtype=torch.float32)
    return data

# Step 4: Run inference
def run_inference(model, sine_wave_data):
    print("Running inference...")
    inputs = prepare_data(sine_wave_data)
    with torch.no_grad():
        outputs = model(inputs)
    return outputs

# Main script
def main():
    seq_length = 100  # Length of each sine wave sequence
    num_samples = 10  # Number of sine wave samples to generate

    # Generate sine wave data
    sine_wave_data = generate_sine_wave(seq_length, num_samples)
    print("Sine wave data:", sine_wave_data)

    # Load model
    model = load_model()

    # Run inference
    predictions = run_inference(model, sine_wave_data)

    print("Predictions:", predictions)

if __name__ == "__main__":
    main()