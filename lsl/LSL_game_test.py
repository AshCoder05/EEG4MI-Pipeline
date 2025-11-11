"""
LSL_game_temp.py – Enhanced Real-Time EEG Classifier
----------------------------------------------------
✅ Loads Stage 1 & Stage 2 models (.joblib)
✅ Connects to EEG LSL stream
✅ Interval-based real-time predictions
✅ Bandpass filtering & feature extraction
✅ Latency tracking
✅ NMCC & ITR metrics
✅ Rolling smoothing for high sensitivity
✅ Ensemble fusion for accuracy boost
"""

import numpy as np
import joblib
import time
from pylsl import StreamInlet, resolve_streams
from scipy.signal import butter, lfilter
from collections import deque
import math

# =====================================================
# ---------------- FILTER DEFINITION ------------------
# =====================================================
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

# =====================================================
# ---------------- CONFIGURATION ----------------------
# =====================================================
MODEL_STAGE1_PATH = "./bci_stage1_rest_intent_iir.joblib"
MODEL_STAGE2_PATH = "./bci_stage2_left_right_iir.joblib"

EEG_STREAM_NAME = "EEGStream"     # Change to your actual stream name
SAMPLE_RATE = 256                 # Hz
INTERVAL_SEC = 2                  # seconds per analysis window
INTERVAL_SAMPLES = SAMPLE_RATE * INTERVAL_SEC
BUFFER_SECONDS = 5
BUFFER_SIZE = SAMPLE_RATE * BUFFER_SECONDS

# =====================================================
# ---------------- LOAD MODELS ------------------------
# =====================================================
print("📂 Loading Stage 1 & Stage 2 models...")
artifact1 = joblib.load(MODEL_STAGE1_PATH)
artifact2 = joblib.load(MODEL_STAGE2_PATH)
print("✅ Models loaded successfully!")

model_stage1 = artifact1["model"] if isinstance(artifact1, dict) else artifact1
model_stage2 = artifact2["model"] if isinstance(artifact2, dict) else artifact2

# =====================================================
# ---------------- CONNECT TO LSL ---------------------
# =====================================================
print(f"🔍 Searching for EEG LSL stream ...")
streams = resolve_streams()
if len(streams) == 0:
    raise RuntimeError("❌ No LSL streams found! Start your EEG stream first.")

eeg_stream = None
for s in streams:
    if EEG_STREAM_NAME.lower() in s.name().lower() or 'eeg' in s.type().lower():
        eeg_stream = s
        break

if eeg_stream is None:
    raise RuntimeError(f"❌ No LSL stream matching '{EEG_STREAM_NAME}' found.")

inlet = StreamInlet(eeg_stream)
print(f"✅ Connected to EEG stream: {eeg_stream.name()} ({eeg_stream.type()})")

# =====================================================
# ---------------- INITIALIZATION ---------------------
# =====================================================
filter_obj = BandpassFilter(lowcut=8, highcut=30, fs=SAMPLE_RATE, order=4)
eeg_buffer = []
pred_history = deque(maxlen=5)

# =====================================================
# ---------------- REAL-TIME LOOP ---------------------
# =====================================================
print("🚀 Starting enhanced real-time classification...\n")

try:
    last_prediction_time = time.time()

    while True:
        # Pull EEG sample
        sample, timestamp = inlet.pull_sample(timeout=1.0)
        if sample is not None:
            eeg_buffer.append(sample)

        # Keep buffer size fixed
        if len(eeg_buffer) > BUFFER_SIZE:
            eeg_buffer = eeg_buffer[-BUFFER_SIZE:]

        # Predict every INTERVAL_SEC
        if len(eeg_buffer) >= INTERVAL_SAMPLES and (time.time() - last_prediction_time >= INTERVAL_SEC):
            last_prediction_time = time.time()
            start_proc = time.time()

            eeg_data = np.array(eeg_buffer[-INTERVAL_SAMPLES:])
            filtered = filter_obj.apply(eeg_data)

            # Extract features
            mean_features = np.mean(filtered, axis=0)
            std_features = np.std(filtered, axis=0)
            features = np.concatenate([mean_features, std_features]).reshape(1, -1)

            # Stage 1 & 2 predictions
            prob1 = model_stage1.predict_proba(features)
            stage1_pred = np.argmax(prob1, axis=1)

            stage2_input = np.concatenate([features, stage1_pred.reshape(-1, 1)], axis=1)
            prob2 = model_stage2.predict_proba(stage2_input)

            # Ensemble fusion for accuracy
            combined_prob = 0.4 * prob1 + 0.6 * prob2
            final_pred = np.argmax(combined_prob, axis=1)

            # Rolling smoothing
            pred_history.append(final_pred[0])
            smooth_pred = max(set(pred_history), key=pred_history.count)

            # Confidence (NMCC)
            nmcc = np.max(combined_prob)

            # Information Transfer Rate (ITR)
            N = combined_prob.shape[1]
            P = nmcc
            T = INTERVAL_SEC
            if P > 0 and P < 1:
                ITR = (math.log2(N) + P*math.log2(P) + (1-P)*math.log2((1-P)/(N-1))) * (60/T)
            else:
                ITR = 0.0

            # Latency
            latency = (time.time() - start_proc) * 1000

            # Print results
            print(f"🧩 Pred: {smooth_pred} | NMCC: {nmcc:.3f} | ITR: {ITR:.2f} bits/min | ⏱ Latency: {latency:.1f} ms")

except KeyboardInterrupt:
    print("\n🛑 Real-time stream stopped by user.")
