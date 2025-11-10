import mne
import time
import os
import joblib 
import numpy as np
from pynput.keyboard import Key, Controller
from pathlib import Path 

# --- Imports required by the loaded model ---
from sklearn.base import BaseEstimator, TransformerMixin
from mne.decoding import CSP
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline, FeatureUnion
from mne.preprocessing import ICA  # <-- Import ICA

# Set MNE to be less "chatty"
mne.set_log_level('WARNING')

# --------------------------
# USER CONFIG
# --------------------------
# --- NEW: Load the IIR (causal) models ---
MODEL_S1_PATH = Path(R"D:\Prsnl\ML\py codes\BCI_Project\src\s8_causal_stage1_model.joblib")
MODEL_S2_PATH = Path(R"D:\Prsnl\ML\py codes\BCI_Project\src\s8_causal_stage2_model.joblib")

# This file MUST be the same type as your calibration data (e.g., 's8')
EDF_FILE_PATH = Path(R"D:\Prsnl\ML\py codes\BCI_Project\src\yaboi\S008_s001_EEG.edf") # The file to simulate

# --- NEW: Real-time simulation settings ---
PREDICTION_INTERVAL_SEC = 0.5 

# --- CONFIG VARIABLES (Must match training) ---
model_channels = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 
    'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2'
] 
notch_freqs = [50.0]
global_sfreq = 160.0 
random_state = 42 # For ICA
causal_filter = True # <-- NEW: Set to True to match training

# --- Initialization ---
keyboard = Controller()

# -----------------------------------------------------------------
# --- CRITICAL "BLUEPRINTS" (with Bug Fix) ---
# -----------------------------------------------------------------
def logvar_transform(X):
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
        
        # --- FIX: Build the dictionary cleanly ---
        filter_params = dict(
            sfreq=self.sfreq, l_freq=self.l_freq, h_freq=self.h_freq,
            method='iir' if self.causal else 'fir',
            iir_params=dict(order=4, ftype='butter', output='sos') if self.causal else None,
            phase='forward' if self.causal else 'zero',
            verbose=False
        )
        
        # --- FIX: Only add 'fir_design' when method is 'fir' ---
        if not self.causal:
            filter_params['fir_design'] = 'firwin'
        # --- END FIX ---
            
        for i in range(X.shape[0]):
            X_filtered[i] = mne.filter.filter_data(X[i], **filter_params)
        return X_filtered

# -----------------------------------------------------------------
# --- PREPROCESSING FUNCTION (CAUSAL + ICA) ---
# This is now identical to your calibration script
# -----------------------------------------------------------------
def load_and_preprocess_with_ica(file_list):
    """
    Loads, renames, and preprocesses custom subject data,
    now including a CAUSAL ICA step.
    """
    if not file_list:
        print("Warning: No files provided to load_and_preprocess_with_ica.")
        return None
        
    try:
        raws = [mne.io.read_raw_edf(f, preload=True, verbose=False) for f in file_list]
    except FileNotFoundError as e:
        print(f"FATAL ERROR: File not found. Did you update the path?")
        print(f"{e}")
        return None
        
    raw = mne.concatenate_raws(raws)
    
    # --- 1. Rename Channels (Custom map for s8 data) ---
    channel_map = {
        'EEG Fp1-Ref': 'Fp1', 'EEG Fp2-Ref': 'Fp2', 'EEG F7-Ref': 'F7',
        'EEG F3-Ref': 'F3', 'EEG Fz-Ref': 'Fz', 'EEG F4-Ref': 'F4',
        'EEG F8-Ref': 'F8', 
        'EEG T3-Ref': 'T7', 'EEG C3-Ref': 'C3', 'EEG Cz-Ref': 'Cz', 'EEG C4-Ref': 'C4',
        'EEG T4-Ref': 'T8', 'EEG T5-Ref': 'P7', 'EEG P3-Ref': 'P3', 'EEG Pz-Ref': 'Pz', 'EEG P4-Ref': 'P4',
        'EEG T6-Ref': 'P8', 'EEG O1-Ref': 'O1', 'EEG O2-Ref': 'O2', 
        'EEG A1-Ref': 'A1', 'EEG A2-Ref': 'A2'
    }
    try:
        raw.rename_channels(channel_map)
    except Exception as e:
        print(f"--- WARNING: Channel renaming failed. (OK if names already match) ---")

    raw.drop_channels(['A1', 'A2'], on_missing='ignore')

    # --- 2. Pick the 19 channels (like in train.py) ---
    available = raw.info['ch_names']
    picks_19 = [ch for ch in model_channels if ch in available]
    if len(picks_19) < 19:
        missing = set(model_channels) - set(available)
        raise RuntimeError(f"Could not find all 19 model channels after renaming. Missing: {missing}")
    raw.pick_channels(picks_19, ordered=False)
    
    raw.set_montage('standard_1005', on_missing='ignore')
    raw.set_eeg_reference('average', projection=False)

    # --- 3. Notch filter *before* ICA (CAUSAL) ---
    print("  Applying CAUSAL notch filter (IIR)...")
    raw.notch_filter(freqs=notch_freqs, method='iir', phase='forward')

    # --- 4. Apply ICA (CAUSAL) ---
    eog_chs = [ch for ch in raw.ch_names if ch.lower() in ('fp1', 'fp2')]
    run_ica = len(eog_chs) >= 2

    if run_ica:
        print(f"  Fitting ICA to remove blinks...")
        # --- FIX: Causal IIR filter for ICA prep ---
        raw_for_ica = raw.copy().filter(l_freq=1.0, h_freq=40.0, method='iir', phase='forward')
        
        n_ica_components = len(raw_for_ica.info['ch_names']) - 1
        ica = ICA(n_components=n_ica_components, max_iter='auto', random_state=random_state)
        ica.fit(raw_for_ica)
        eog_indices, _ = ica.find_bads_eog(raw, ch_name=eog_chs)
        print(f"  ICA found {len(eog_indices)} EOG components to remove.")
        ica.exclude = eog_indices
        ica.apply(raw) # Clean the raw data
    else:
        print(f"  Skipping ICA (FP1/FP2 missing).")

    # --- 5. Resample (after all other steps) ---
    if raw.info['sfreq'] != global_sfreq:
        print(f"Resampling data from {raw.info['sfreq']}Hz to {global_sfreq}Hz...")
        raw.resample(global_sfreq)
        
    return raw

# -----------------------------------------------------------------
# --- Real-Time Sliding Window Function (WITH VALIDATION) ---
# -----------------------------------------------------------------
def run_bci_simulation_sliding(model_s1, model_s2, meta_s1, meta_s2, raw, interval_sec):
    """
    Simulates a REAL-TIME BCI by continuously
    predicting on a sliding window.
    Also checks annotations to print the "Ground Truth" label.
    """
    print("\n--- Starting BCI Simulation Loop (Sliding Window) ---")
    
    # Get all metadata
    sfreq = meta_s1['sfreq']
    feature_tmin, feature_tmax = meta_s1['feature_window']
    class_map_s1 = meta_s1['class_map']
    class_map_s2 = meta_s2['class_map']
    
    # Calculate window size in seconds and samples
    window_size_sec = feature_tmax - feature_tmin
    window_size_samples = int(window_size_sec * sfreq)
    
    # Get total file duration
    total_duration_sec = raw.times[-1]
    
    # --- Set how long a task marker (like 'left') lasts ---
    TASK_DURATION_SEC = 4.0 
    
    print(f"Config: Window Size={window_size_sec:.2f}s, Prediction Interval={interval_sec:.2f}s")
    
    pressed_keys = set()
    current_time_sec = 0.0

    # --- This loop is the real-time "game loop" ---
    while current_time_sec + window_size_sec < total_duration_sec:
        
        loop_start_time = time.time()
        final_action = 'IDLE' # Default state
        
        try:
            # 1. Calculate window boundaries
            win_start_samp = int(current_time_sec * sfreq)
            win_stop_samp = win_start_samp + window_size_samples
            
            # 2. Extract data chunk from the *pre-cleaned* raw file
            data = raw.get_data(picks=meta_s1['channels'], start=win_start_samp, stop=win_stop_samp)

            # 3. Reshape for model
            if data.shape[1] == window_size_samples:
                X = data[np.newaxis, :, :] # Shape: (1, 19, 320)
                
                # --- STAGE 1 PREDICTION (THE "BRAIN BRAKE") ---
                pred_s1 = model_s1.predict(X)[0]
                label_s1 = class_map_s1[str(pred_s1)] 

                if label_s1 == 'INTENT':
                    # --- STAGE 2 PREDICTION (THE "DIRECTION") ---
                    pred_s2 = model_s2.predict(X)[0]
                    label_s2 = class_map_s2[str(pred_s2)]
                    final_action = label_s2.upper() # 'LEFT' or 'RIGHT'
                else:
                    final_action = 'REST'
            
            else:
                final_action = 'REST' # Treat end-of-file as rest
                
        except Exception as e:
            print(f"Error during prediction at {current_time_sec:.2f}s: {e}")
            final_action = 'REST' # Fail-safe

        # --- VALIDATION LOGIC ---
        # Check the annotations to find the "Ground Truth"
        true_label = "REST" # Assume REST by default
        for onset, duration, desc in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description):
            task_start = onset
            task_end = onset + TASK_DURATION_SEC # Assume task lasts 4s
            
            # Check if our *current time* is inside a task window
            if current_time_sec >= task_start and current_time_sec < task_end:
                desc_lower = desc.lower()
                if 'left' in desc_lower:
                    true_label = "LEFT"
                    break
                elif 'right' in desc_lower:
                    true_label = "RIGHT"
                    break
        # --- END VALIDATION LOGIC ---

        # Print the side-by-side comparison
        print(f"[{current_time_sec:7.2f}s / {total_duration_sec:7.2f}s] -> TRUE: {true_label:<5} -> Predicted: {final_action}")
        
        # --- Keyboard control ---
        if final_action == 'LEFT':
            if Key.right in pressed_keys: keyboard.release(Key.right); pressed_keys.remove(Key.right)
            if Key.left not in pressed_keys: keyboard.press(Key.left); pressed_keys.add(Key.left)
        elif final_action == 'RIGHT':
            if Key.left in pressed_keys: keyboard.release(Key.left); pressed_keys.remove(Key.left)
            if Key.right not in pressed_keys: keyboard.press(Key.right); pressed_keys.add(Key.right)
        elif final_action == 'REST':
            for k in list(pressed_keys):
                keyboard.release(k)
                pressed_keys.remove(k)
        
        # --- Simulate Real-Time ---
        current_time_sec += interval_sec
        processing_time = time.time() - loop_start_time
        time_to_wait = interval_sec - processing_time
        if time_to_wait > 0:
            time.sleep(time_to_wait)
            
    # 5. Clean up
    print("\n--- BCI Simulation Finished ---")
    for k in list(pressed_keys):
        keyboard.release(k)
        pressed_keys.remove(k)

# --- Main execution ---
if __name__ == "__main__":
    
    if not (MODEL_S1_PATH.exists() and MODEL_S2_PATH.exists()):
        print(f"Error: Model files not found.")
        exit()
    if not EDF_FILE_PATH.exists():
        print(f"Error: EDF file not found at {EDF_FILE_PATH}")
        exit()

    print(f"Loading Stage 1 Model (Rest/Intent)...")
    try:
        artifact_s1 = joblib.load(MODEL_S1_PATH)
        model_s1 = artifact_s1['model']
        meta_s1 = artifact_s1['meta'] 
        
        print(f"Loading Stage 2 Model (Left/Right)...")
        artifact_s2 = joblib.load(MODEL_S2_PATH)
        model_s2 = artifact_s2['model']
        meta_s2 = artifact_s2['meta']
        
        print("Models and metadata loaded successfully.")
        
        # --- Check for filter mismatch ---
        if meta_s1['causal'] != causal_filter:
            print("="*80)
            print("FATAL ERROR: FILTER MISMATCH")
            print(f"Models were trained with causal_filter = {meta_s1['causal']}")
            print(f"This script is set to causal_filter = {causal_filter}")
            print("You MUST set the 'causal_filter' variable in this script to match the model.")
            print("="*80)
            exit()
    
        print(f"Loading and preprocessing EDF file (this may take a moment)...")
        
        # --- This function now loads, filters, and applies ICA to the whole file ---
        raw = load_and_preprocess_with_ica([EDF_FILE_PATH])
        
        if raw is None:
            raise RuntimeError("Failed to load and preprocess EDF file.")
        
        print("Preprocessing complete.")
        
        # 3. Run the BCI controller
        print("\nFocus on your game window! Simulation starts in 5 seconds...")
        time.sleep(5)
        # --- Call the NEW sliding window function ---
        run_bci_simulation_sliding(model_s1, model_s2, meta_s1, meta_s2, raw, PREDICTION_INTERVAL_SEC)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()