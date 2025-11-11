import joblib
import time
import numpy as np
import mne
from pathlib import Path 
from pylsl import StreamInlet, resolve_streams
from pynput.keyboard import Key, Controller
from scipy.signal import butter, sosfilt, sosfilt_zi # For IIR/Notch filtering

# --- Imports required by the loaded model ---
from sklearn.base import BaseEstimator, TransformerMixin
from mne.decoding import CSP
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline, FeatureUnion
from mne.preprocessing import ICA 

# Set MNE to be less "chatty"
mne.set_log_level('WARNING')

# ==========================
# ---- CONFIGURATION --------
# ==========================
# --- Paths to your CALIBRATED, CAUSAL, ICA-based models ---
MODEL_S1_PATH = Path(R"D:\Prsnl\ML\py codes\BCI_Project\src\s8_causal_stage1_model.joblib")
MODEL_S2_PATH = Path(R"D:\Prsnl\ML\py codes\BCI_Project\src\s8_causal_stage2_model.joblib")

# --- LSL & Prediction Settings ---
EEG_STREAM_NAME = "OpenBCI_EEG"   # <--- Change this to your LSL Stream Name!
PREDICTION_INTERVAL_SEC = 0.25    # Predict 4 times per second
BUFFER_WINDOW_SEC = 2.0           # Must match feature_tmax - feature_tmin (0, 2)

# --- Model Config (Must match training!) ---
# We will load the *actual* channel list from the model metadata
notch_freqs = [50.0]
global_sfreq = 160.0 # This will be verified against the model's metadata
causal_filter = True # This script is built for causal models

# --- Initialization ---
keyboard = Controller()
pressed_keys = set() # Keep track of which keys are down

# ==========================
# ---- HELPER CLASSES --------
# ==========================

# --- Re-include Blueprints for joblib loading ---
def logvar_transform(X):
    var = X.var(axis=2)
    return np.log(var + 1e-10)

class BandpassFilter(BaseEstimator, TransformerMixin):
    # This class is only needed for joblib to load the model pipeline
    def __init__(self, l_freq, h_freq, sfreq, causal=True):
        self.l_freq, self.h_freq, self.sfreq, self.causal = l_freq, h_freq, sfreq, causal
    def fit(self, X, y=None): return self
    def transform(self, X):
        # This transform is part of the FBCSP pipeline and should not be called directly
        raise NotImplementedError("BandpassFilter is part of the FBCSP pipeline.")
        
# --- Real-Time Notch Filter State Manager ---
class CausalNotchFilter:
    """Manages the state (zi) for a causal IIR Notch filter."""
    def __init__(self, freqs, sfreq, n_channels, order=4):
        self.freqs = freqs
        self.sfreq = sfreq
        self.order = order
        self.sos_filters = []
        self.zi_states = []
        
        for f in self.freqs:
            q = 30.0 # Quality factor
            sos = butter(self.order, [f-f/q, f+f/q], btype='bandstop', output='sos', fs=self.sfreq)
            self.sos_filters.append(sos)
            # Initialize zi state for n_channels
            zi_init = sosfilt_zi(sos) # Shape (n_sections, 2)
            # We need (n_sections, n_channels, 2)
            zi_shape = (sos.shape[0], n_channels, 2)
            self.zi_states.append(np.broadcast_to(zi_init[:, np.newaxis, :], zi_shape).copy())
            
    def apply(self, data):
        """
        Applies the notch filter to a new chunk of data.
        Data shape: (n_channels, n_samples)
        """
        filtered_data = data.copy()
        for i, sos in enumerate(self.sos_filters):
            # sosfilt and update the filter state (zi)
            filtered_data, self.zi_states[i] = sosfilt(sos, filtered_data, axis=1, zi=self.zi_states[i])
        return filtered_data

# ==========================
# ---- MAIN EXECUTION -------
# ==========================
def main_bci_live():
    
    # 1. Load Models and Metadata
    print("📂 Loading Stage 1 & Stage 2 models...")
    try:
        artifact_s1 = joblib.load(MODEL_S1_PATH)
        model_s1 = artifact_s1['model']
        meta_s1 = artifact_s1['meta'] 
        
        artifact_s2 = joblib.load(MODEL_S2_PATH)
        model_s2 = artifact_s2['model']
        meta_s2 = artifact_s2['meta']

        # --- Get config from metadata ---
        sfreq = meta_s1['sfreq']
        model_channels = meta_s1['channels'] 
        n_channels = len(model_channels)
        
        # Check that model is causal
        if not meta_s1['causal']:
            raise RuntimeError("FATAL ERROR: Loaded model is NOT Causal (FIR). This script requires a Causal (IIR) model.")
        if int(sfreq) != int(global_sfreq):
             raise RuntimeError(f"Model sfreq ({sfreq}Hz) does not match script ({global_sfreq}Hz).")

        # --- ** LOAD THE FITTED ICA (THE "KEY") ** ---
        print("   Loading pre-fitted ICA object...")
        ica_obj = meta_s1.get('ica_obj')
        
        if ica_obj:
            # We don't need the whole object, just the pre-fitted matrices
            ica_unmixing = ica_obj.unmixing_matrix_
            ica_mixing = ica_obj.mixing_matrix_
            ica_exclude = ica_obj.exclude
            run_ica = True
            print(f"   ICA is ENABLED (will remove {len(ica_exclude)} components).")
        else:
            run_ica = False
            print("   WARNING: ICA is DISABLED (ica_obj not found in model file).")

        # --- *** CORRECTED: LOAD BOTH EXTRACTORS AND CLASSIFIERS *** ---
        print("   Extracting feature pipelines and classifiers...")
        feature_extractor_s1 = Pipeline(model_s1.steps[:-1])
        classifier_s1 = model_s1.steps[-1][1]
        
        feature_extractor_s2 = Pipeline(model_s2.steps[:-1])
        classifier_s2 = model_s2.steps[-1][1]
        # --- *** END FIX *** ---
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. Connect to LSL Stream
    print(f"🔍 Looking for EEG LSL stream (name containing '{EEG_STREAM_NAME}')...")
    streams = resolve_streams(timeout=3.0)
    
    inlet = None
    for s in streams:
        if EEG_STREAM_NAME.lower() in s.name().lower() or 'eeg' in s.type().lower():
            inlet = StreamInlet(s)
            print(f"✅ Connected to EEG stream: {s.name()}")
            break
            
    if inlet is None:
        raise RuntimeError(f"❌ No LSL stream matching name '{EEG_STREAM_NAME}' found.")

    # --- LSL Channel Verification ---
    info = inlet.info()
    ch_names_lsl = [ch.child_value("label") for ch in info.desc().child("channels").children("channel")]
    
    # Find the indices of the channels our model needs
    model_indices = []
    try:
        for ch_name in model_channels:
            model_indices.append(ch_names_lsl.index(ch_name))
    except ValueError as e:
        raise RuntimeError(f"FATAL: Stream is missing channel '{e}' needed by the model. Check your headset/LSL setup.")

    print(f"   Stream has {len(ch_names_lsl)} channels. Model will use {len(model_indices)} channels.")

    # 3. Initialize Real-Time Components
    window_size_samples = int(BUFFER_WINDOW_SEC * sfreq)
    eeg_buffer = np.zeros((n_channels, 0)) 
    notch_filter = CausalNotchFilter(freqs=notch_freqs, sfreq=sfreq, n_channels=n_channels)
    last_pred_time = time.time()
    
    print(f"\n🚀 Starting real-time classification (Window: {BUFFER_WINDOW_SEC}s, Interval: {PREDICTION_INTERVAL_SEC}s)..")
    print("Focus on your game window. Press Ctrl+C to stop.")
    
    # 4. Main Real-Time Loop
    try:
        while True:
            # --- Pull all available samples (Chunking) ---
            chunk, timestamps = inlet.pull_chunk(timeout=0.1, max_samples=int(sfreq * PREDICTION_INTERVAL_SEC))
            
            if chunk:
                # Get data, transpose to (channels, samples)
                chunk_data_all = np.array(chunk).T
                # Select *only* the channels we need, in the correct order
                chunk_data = chunk_data_all[model_indices, :]

                # ** 1. APPLY CAUSAL NOTCH FILTER **
                chunk_data_notched = notch_filter.apply(chunk_data)

                # ** 2. APPLY CAUSAL ICA (THE "KEY") **
                cleaned_chunk = chunk_data_notched
                if run_ica:
                    # This is the fast, real-time ICA.apply()
                    unmixed_data = ica_unmixing @ chunk_data_notched
                    unmixed_data[ica_exclude, :] = 0
                    cleaned_chunk = ica_mixing @ unmixed_data

                # ** 3. APPEND TO BUFFER **
                eeg_buffer = np.concatenate([eeg_buffer, cleaned_chunk], axis=1)

            # --- Check Prediction Interval ---
            current_time = time.time()
            if current_time - last_pred_time >= PREDICTION_INTERVAL_SEC:
                
                if eeg_buffer.shape[1] >= window_size_samples:
                    
                    # 1. Take the *newest* chunk of data from the buffer
                    data_window = eeg_buffer[:, -window_size_samples:] 
                    
                    # 2. Reshape to (n_epochs, n_channels, n_times)
                    X = data_window[np.newaxis, :, :] # Shape: (1, 19, 320)
                    
                    # --- *** CORRECTED: RUN STAGE 1 (Extractor + Classifier) *** ---
                    features_s1 = feature_extractor_s1.transform(X)
                    pred_s1 = classifier_s1.predict(features_s1)[0]
                    label_s1 = meta_s1['class_map'].get(str(pred_s1), 'UNKNOWN')
                    
                    final_action = 'IDLE'
                    if label_s1 == 'INTENT':
                        # --- *** CORRECTED: RUN STAGE 2 (Extractor + Classifier) *** ---
                        features_s2 = feature_extractor_s2.transform(X)
                        pred_s2 = classifier_s2.predict(features_s2)[0]
                        label_s2 = meta_s2['class_map'].get(str(pred_s2), 'UNKNOWN')
                        final_action = label_s2.upper()
                    else:
                        final_action = 'REST'
                    # --- *** END FIX *** ---
                        
                    print(f"[{time.strftime('%H:%M:%S')}] STAGE1: {label_s1:<6} | FINAL: {final_action}")
                    
                    if final_action == 'LEFT':
                        if Key.right in pressed_keys: keyboard.release(Key.right); pressed_keys.remove(Key.right)
                        if Key.left not in pressed_keys: keyboard.press(Key.left); pressed_keys.add(Key.left)
                    elif final_action == 'RIGHT':
                        if Key.left in pressed_keys: keyboard.release(Key.left); pressed_keys.remove(Key.left)
                        if Key.right not in pressed_keys: keyboard.press(Key.right); pressed_keys.add(Key.right)
                    elif final_action == 'REST':
                        for k in list(pressed_keys): keyboard.release(k); pressed_keys.remove(k)
                        
                    last_pred_time = current_time

                    # 4. Trim buffer to prevent memory leak
                    # Keep 2x the window size, just to be safe
                    eeg_buffer = eeg_buffer[:, -(window_size_samples * 2):] 
                
                else:
                    # Not enough data yet
                    print(f"Waiting for buffer to fill ({eeg_buffer.shape[1]}/{window_size_samples} samples)...", end='\r')
            
            # Prevent 100% CPU usage
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n🛑 Real-time stream stopped by user.")
    finally:
        # Clean up
        for k in list(pressed_keys): keyboard.release(k); pressed_keys.remove(k)
        if inlet:
            inlet.close_stream()
        print("Stream closed. Goodbye.")

if __name__ == "__main__":
    main_bci_live()