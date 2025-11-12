#!/usr/bin/env python3
"""
bci_live_predict_FIXED.py

Real-time BCI prediction script with two critical fixes:
1. Correctly maps LSL channel names to model channel indices.
2. Applies a manual average reference to incoming data chunks.
"""

import joblib
import time
import numpy as np
import mne
from pathlib import Path
from pylsl import StreamInlet, resolve_streams, resolve_byprop
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
MODEL_S1_PATH = Path(R"C:\Users\sujit\OneDrive\Desktop\Mini Project\models\s11_causal_stage1_model.joblib")
MODEL_S2_PATH = Path(R"C:\Users\sujit\OneDrive\Desktop\Mini Project\models\s11_causal_stage2_model.joblib")

# --- LSL & Prediction Settings ---
EEG_STREAM_NAME = "OpenBCI_EEG" # Used for reference, but we search by type 'EEG'
PREDICTION_INTERVAL_SEC = 0.25
BUFFER_WINDOW_SEC = 2.0 # This must be <= (feature_tmax - feature_tmin) from calibration

# --- Model Config (Must match training!) ---
notch_freqs = [50.0]
global_sfreq = 160.0 
causal_filter = True 

# --- Initialization ---
keyboard = Controller()
pressed_keys = set() 

# ==========================
# ---- HELPER CLASSES --------
# ==========================

# --- Re-include Blueprints for joblib loading ---
def logvar_transform(X):
    """Log-variance feature extraction, typically used after CSP."""
    var = X.var(axis=2)
    return np.log(var + 1e-10)

class BandpassFilter(BaseEstimator, TransformerMixin):
    """A scikit-learn compatible MNE bandpass filter."""
    def __init__(self, l_freq, h_freq, sfreq, causal=True):
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.sfreq = sfreq
        self.causal = causal
    def fit(self, X, y=None): return self
    def transform(self, X):
        """X shape is (n_epochs, n_channels, n_times)"""
        X_filtered = np.zeros_like(X)
        
        # Build the MNE filter parameters
        filter_params = dict(
            sfreq=self.sfreq, l_freq=self.l_freq, h_freq=self.h_freq,
            method='iir' if self.causal else 'fir',
            iir_params=dict(order=4, ftype='butter', output='sos') if self.causal else None,
            phase='forward' if self.causal else 'zero',
            verbose=False
        )
        
        if not self.causal:
            filter_params['fir_design'] = 'firwin'
            
        # Apply the filter to each epoch (in this case, just the one)
        for i in range(X.shape[0]):
            X_filtered[i] = mne.filter.filter_data(X[i], **filter_params)
        return X_filtered
        
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
            
            zi_init = sosfilt_zi(sos) 
            zi_shape = (sos.shape[0], n_channels, 2)
            self.zi_states.append(np.broadcast_to(zi_init[:, np.newaxis, :], zi_shape).copy())
            
    def apply(self, data):
        """
        Applies the notch filter to a new chunk of data, updating the internal state.
        Data shape: (n_channels, n_samples)
        """
        filtered_data = data.copy()
        for i, sos in enumerate(self.sos_filters):
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
        model_channels = meta_s1['channels'] # The 19 channels
        n_channels = len(model_channels) 
        
        if not meta_s1['causal']:
            raise RuntimeError("FATAL ERROR: Loaded model is NOT Causal (FIR).")
        if int(sfreq) != int(global_sfreq):
             raise RuntimeError(f"Model sfreq ({sfreq}Hz) does not match script ({global_sfreq}Hz).")

        # --- ** LOAD THE FITTED ICA (THE "KEY") ** ---
        print("    Loading pre-fitted ICA object...")
        ica_obj = meta_s1.get('ica_obj')
        
        if ica_obj:
            ica_unmixing = ica_obj.unmixing_matrix_
            ica_mixing = ica_obj.mixing_matrix_
            ica_exclude = ica_obj.exclude
            run_ica = True
            print(f"    ICA is ENABLED (will remove {len(ica_exclude)} components).")
        else:
            run_ica = False
            print("    WARNING: ICA is DISABLED (ica_obj not found in model file).")

        # --- Extract both feature pipelines and classifiers ---
        print("    Extracting feature pipelines and classifiers...")
        feature_extractor_s1 = Pipeline(model_s1.steps[:-1])
        classifier_s1 = model_s1.steps[-1][1]
        
        feature_extractor_s2 = Pipeline(model_s2.steps[:-1])
        classifier_s2 = model_s2.steps[-1][1]
        
    except FileNotFoundError as e:
        print(f"❌ FATAL ERROR: Model file not found.")
        print(f"    Check paths in CONFIG section (lines 31-32).")
        print(f"    Error details: {e}")
        return
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. Connect to LSL Stream
    print(f"🔍 Looking for EEG LSL stream (type 'EEG')...")
    
    streams = resolve_byprop('type', 'EEG', timeout=5.0)
    
    if not streams:
        raise RuntimeError(f"❌ No LSL stream matching type 'EEG' found after 5.0 seconds. Check your acquisition software and firewall.")
        
    inlet = StreamInlet(streams[0])
    print(f"✅ Connected to EEG stream: {streams[0].name()}")
    
    # --- *** NEW: LSL CHANNEL MAPPING & VALIDATION (THE FIRST FIX) *** ---
    
    # This is the "secret handshake" map from your calibration script
    channel_map = {
        'EEG Fp1-Ref': 'Fp1', 'EEG Fp2-Ref': 'Fp2', 'EEG F7-Ref': 'F7',
        'EEG F3-Ref': 'F3', 'EEG Fz-Ref': 'Fz', 'EEG F4-Ref': 'F4',
        'EEG F8-Ref': 'F8', 
        'EEG T3-Ref': 'T7', 'EEG C3-Ref': 'C3', 'EEG Cz-Ref': 'Cz', 'EEG C4-Ref': 'C4',
        'EEG T4-Ref': 'T8', 'EEG T5-Ref': 'P7', 'EEG P3-Ref': 'P3', 'EEG Pz-Ref': 'Pz', 'EEG P4-Ref': 'P4',
        'EEG T6-Ref': 'P8', 'EEG O1-Ref': 'O1', 'EEG O2-Ref': 'O2', 
        'EEG A1-Ref': 'A1', 'EEG A2-Ref': 'A2'
    }

    info = inlet.info()
    ch_count_lsl = info.channel_count()
    
    # Get all channel names from the LSL stream
    chlist = info.desc().child("channels").child("channel")
    lsl_ch_names = []
    for i in range(ch_count_lsl):
        lsl_ch_names.append(chlist.child_value("label"))
        chlist = chlist.next_sibling()

    # Rename them in our list using the map
    lsl_ch_names_renamed = [channel_map.get(ch, ch) for ch in lsl_ch_names]
    print(f"    Found {ch_count_lsl} LSL channels. Renamed list: {lsl_ch_names_renamed}")

    # Now, find the INDICES of our model channels in this renamed list
    try:
        model_indices = [lsl_ch_names_renamed.index(ch) for ch in model_channels]
    except ValueError as e:
        # This error means a channel in 'model_channels' was not found in the LSL stream
        missing_ch = str(e).split("'")[1]
        print(f"❌ FATAL: Could not find model channel '{missing_ch}' in LSL stream.")
        print(f"    Model needs: {model_channels}")
        print(f"    LSL stream (renamed) has: {lsl_ch_names_renamed}")
        return

    print(f"    Successfully mapped {len(model_indices)} model channels to LSL indices.")
    # 'model_indices' is now the list of correct indices to pick, e.g., [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    # (assuming they are in order, but it works even if they aren't)
    
    if len(model_indices) != n_channels:
         raise RuntimeError(f"FATAL: Logic error. Mapped {len(model_indices)} indices, but model expects {n_channels}.")
    # --- *** END OF NEW MAPPING CODE *** ---


    # 3. Initialize Real-Time Components
    window_size_samples = int(BUFFER_WINDOW_SEC * sfreq)
    eeg_buffer = np.zeros((n_channels, 0))
    # This filter is correctly initialized for 19 channels
    notch_filter = CausalNotchFilter(freqs=notch_freqs, sfreq=sfreq, n_channels=n_channels)
    last_pred_time = time.time()
    
    print(f"\n🚀 Starting real-time classification (Window: {BUFFER_WINDOW_SEC}s, Interval: {PREDICTION_INTERVAL_SEC}s)..")
    print("Focus on your game window. Press Ctrl+C to stop.")
    
    # 4. Main Real-Time Loop
    try:
        while True:
            chunk, timestamps = inlet.pull_chunk(timeout=0.1, max_samples=int(sfreq * PREDICTION_INTERVAL_SEC))
            
            if chunk:
                chunk_data_all = np.array(chunk).T
                
                # --- *** FIX: FULL PREPROCESSING PIPELINE (THE SECOND FIX) *** ---
                
                # ** 1. PICK (This now uses the *correct* indices from Section 2) **
                chunk_data_picked = chunk_data_all[model_indices, :] 

                # ** 2. MANUAL AVERAGE REFERENCE (THE MISSING STEP) **
                # (n_chans, n_samples) - (1, n_samples) -> (n_chans, n_samples)
                chunk_data_ref = chunk_data_picked - np.mean(chunk_data_picked, axis=0, keepdims=True)

                # ** 3. APPLY CAUSAL NOTCH FILTER **
                # Pass the *re-referenced* data to the filter
                chunk_data_notched = notch_filter.apply(chunk_data_ref)

                # ** 4. APPLY CAUSAL ICA (THE "KEY") **
                cleaned_chunk = chunk_data_notched
                if run_ica:
                    # This will now be (N_comps, 19) @ (19, N_samples) -> SUCCESS
                    # And it's operating on re-referenced data, as it was trained to.
                    unmixed_data = ica_unmixing @ chunk_data_notched 
                    unmixed_data[ica_exclude, :] = 0
                    cleaned_chunk = ica_mixing @ unmixed_data

                # ** 5. APPEND TO BUFFER **
                eeg_buffer = np.concatenate([eeg_buffer, cleaned_chunk], axis=1)
                
                # --- *** END OF FIX *** ---

            # --- Check Prediction Interval ---
            current_time = time.time()
            if current_time - last_pred_time >= PREDICTION_INTERVAL_SEC:
                
                if eeg_buffer.shape[1] >= window_size_samples:
                    
                    # Get the most recent window of data
                    data_window = eeg_buffer[:, -window_size_samples:]
                    
                    # Shape: (1, 19, N_samples)
                    X = data_window[np.newaxis, :, :] 
                    
                    # --- RUN STAGE 1 (Extractor + Classifier) ---
                    # This will now work: 19-channel CSP receives 19-channel data
                    features_s1 = feature_extractor_s1.transform(X) 
                    pred_s1 = classifier_s1.predict(features_s1)[0]
                    label_s1 = meta_s1['class_map'].get(str(pred_s1), 'UNKNOWN')
                    
                    final_action = 'IDLE'
                    if label_s1 == 'INTENT':
                        # --- RUN STAGE 2 (Extractor + Classifier) ---
                        features_s2 = feature_extractor_s2.transform(X)
                        pred_s2 = classifier_s2.predict(features_s2)[0]
                        label_s2 = meta_s2['class_map'].get(str(pred_s2), 'UNKNOWN')
                        final_action = label_s2.upper()
                    else:
                        final_action = 'REST'
                        
                    print(f"[{time.strftime('%H:%M:%S')}] STAGE1: {label_s1:<6} | FINAL: {final_action}")
                    
                    # --- Keyboard Control ---
                    if final_action == 'LEFT':
                        if Key.right in pressed_keys: keyboard.release(Key.right); pressed_keys.remove(Key.right)
                        if Key.left not in pressed_keys: keyboard.press(Key.left); pressed_keys.add(Key.left)
                    elif final_action == 'RIGHT':
                        if Key.left in pressed_keys: keyboard.release(Key.left); pressed_keys.remove(Key.left)
                        if Key.right not in pressed_keys: keyboard.press(Key.right); pressed_keys.add(Key.right)
                    elif final_action == 'REST':
                        for k in list(pressed_keys): keyboard.release(k); pressed_keys.remove(k)
                        
                    last_pred_time = current_time

                    # 4. Trim buffer to prevent memory leak (keep 2x window size)
                    eeg_buffer = eeg_buffer[:, -(window_size_samples * 2):]
                
                else:
                    # Not enough data yet
                    print(f"Waiting for buffer to fill ({eeg_buffer.shape[1]}/{window_size_samples} samples)...", end='\r')
            
            # Prevent 100% CPU usage
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n🛑 Real-time stream stopped by user.")
    finally:
        # Clean up: release any pressed keys and close LSL stream
        for k in list(pressed_keys): keyboard.release(k); pressed_keys.remove(k)
        if 'inlet' in locals() and inlet:
            inlet.close_stream()
        print("Stream closed. Goodbye.")

if __name__ == "__main__":
    main_bci_live()
