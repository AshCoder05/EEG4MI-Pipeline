"""
LSL_game.py - Real-time EEG classifier using pre-trained Stage 1 & Stage 2 models.

✅ Features:
- Loads .joblib models (Stage 1 & Stage 2)
- Defines missing custom classes used during training (BandpassFilter, etc.)
- Connects to LSL EEG stream
- Applies same preprocessing (butter filter, normalization)
- Predicts live every N seconds
"""

import numpy as np
import joblib
import time
from pylsl import StreamInlet, resolve_streams
from scipy.signal import butter, lfilter

# ==========================
# ---- FILTER DEFINITIONS ---
# ==========================
class BandpassFilter:
    def __init__(self, lowcut=8, highcut=30, fs=256, order=5):
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order
        self.b, self.a = butter(order, [lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype='band')

    def apply(self, data):
        if len(data) < self.order:
            return data
        return lfilter(self.b, self.a, data, axis=0)

# ==========================
# ---- CONFIGURATION --------
# ==========================
MODEL_STAGE1_PATH = "./bci_stage1_rest_intent_iir.joblib"
MODEL_STAGE2_PATH = "./bci_stage2_left_right_iir.joblib"

EEG_STREAM_NAME = "EEGStream"   # Change to your actual LSL stream name
SAMPLE_RATE = 256               # Hz
BUFFER_SECONDS = 5
BUFFER_SIZE = SAMPLE_RATE * BUFFER_SECONDS

# ==========================
# ---- LOAD MODELS ----------
# ==========================
print("📂 Loading Stage 1 & Stage 2 models...")
artifact1 = joblib.load(MODEL_STAGE1_PATH)
artifact2 = joblib.load(MODEL_STAGE2_PATH)

print("✅ Models loaded successfully!")

# Extract pipeline stages (if stored as dicts)
model_stage1 = artifact1["model"] if isinstance(artifact1, dict) else artifact1
model_stage2 = artifact2["model"] if isinstance(artifact2, dict) else artifact2

# ==========================
# ---- CONNECT TO LSL -------
# ==========================
print(f"🔍 Looking for EEG LSL streams ...")
streams = resolve_streams()  # ✅ works in your pylsl version
if len(streams) == 0:
    raise RuntimeError("❌ No LSL streams found! Make sure your EEG device is streaming.")

# Find the EEG stream by name or type
eeg_stream = None
for s in streams:
    if EEG_STREAM_NAME.lower() in s.name().lower() or 'eeg' in s.type().lower():
        eeg_stream = s
        break

if eeg_stream is None:
    raise RuntimeError(f"❌ No LSL stream matching name '{EEG_STREAM_NAME}' found among {len(streams)} streams.")

inlet = StreamInlet(eeg_stream)
print(f"✅ Connected to EEG stream: {eeg_stream.name()} ({eeg_stream.type()})")

# ==========================
# ---- BUFFER INITIALIZE ----
# ==========================
eeg_buffer = []
filter_obj = BandpassFilter(lowcut=8, highcut=30, fs=SAMPLE_RATE, order=4)

# ==========================
# ---- REAL-TIME LOOP -------
# ==========================
print("🚀 Starting real-time classification...\n")

try:
    while True:
        # Pull one sample at a time
        sample, timestamp = inlet.pull_sample(timeout=1.0)
        if sample is not None:
            eeg_buffer.append(sample)

        # Keep buffer length fixed
        if len(eeg_buffer) > BUFFER_SIZE:
            eeg_buffer = eeg_buffer[-BUFFER_SIZE:]

        # Run prediction every BUFFER_SECONDS
        if len(eeg_buffer) >= BUFFER_SIZE:
            eeg_data = np.array(eeg_buffer)

            # Apply filtering
            filtered = filter_obj.apply(eeg_data)

            # Feature extraction (mean, std per channel)
            mean_features = np.mean(filtered, axis=0)
            std_features = np.std(filtered, axis=0)
            features = np.concatenate([mean_features, std_features]).reshape(1, -1)

            # Stage 1 Prediction
            stage1_pred = model_stage1.predict(features)

            # Stage 2 Prediction (optional refinement)
            final_pred = model_stage2.predict(np.concatenate([features, stage1_pred.reshape(-1, 1)], axis=1))

            # Print live output
            print(f"[🧩] Stage1: {stage1_pred[0]} | Stage2: {final_pred[0]} | Buffer: {len(eeg_buffer)} samples")

            # Wait a bit before next inference
            time.sleep(BUFFER_SECONDS / 2)

except KeyboardInterrupt:
    print("\n🛑 Real-time stream stopped by user.")
