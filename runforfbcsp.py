import mne
import time
import os
import joblib 
import numpy as np
from pynput.keyboard import Key, Controller
from pathlib import Path # <-- ADDED

# --- ADDED: Imports required by the loaded model ---
from sklearn.base import BaseEstimator, TransformerMixin
from mne.decoding import CSP
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline, FeatureUnion
from mne.preprocessing import ICA

# Set MNE to be less "chatty"
mne.set_log_level('WARNING')

# --- Configuration ---
# FIXED (Fix 7): Use pathlib.Path for robust paths
MODEL_S1_PATH = Path(R'D:\Prsnl\ML\py codes\BCI_Project\src\bci_stage1_rest_intent_model.joblib')
MODEL_S2_PATH = Path(R'D:\Prsnl\ML\py codes\BCI_Project\src\bci_stage2_left_right_model.joblib')
EDF_FILE_PATH = Path(R'D:\Prsnl\ML\py codes\BCI_Project\S004R04.edf') # The file to simulate

# --- Initialization ---
keyboard = Controller()

# -----------------------------------------------------------------
# --- MISSING "BLUEPRINTS" (Added to fix the load error) ---
# These custom classes *must* be defined for joblib to load the model.
# -----------------------------------------------------------------
def logvar_transform(X):
    """X: (n_epochs, n_components, n_times) -> (n_epochs, n_components)"""
    var = X.var(axis=2)
    return np.log(var + 1e-10)

class BandpassFilter(BaseEstimator, TransformerMixin):
    """A scikit-learn compatible MNE bandpass filter."""
    def __init__(self, l_freq, h_freq, sfreq, causal=False):
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.sfreq = sfreq
        self.causal = causal

    def fit(self, X, y=None):
        return self # No fitting needed

    def transform(self, X):
        """X shape is (n_epochs, n_channels, n_times)"""
        X_filtered = np.zeros_like(X)
        filter_params = dict(
            sfreq=self.sfreq, l_freq=self.l_freq, h_freq=self.h_freq,
            method='iir' if self.causal else 'fir',
            iir_params=dict(order=4, ftype='butter', output='sos') if self.causal else None,
            phase='forward' if self.causal else 'zero', 
            fir_design='firwin' if not self.causal else None,
            verbose=False 
        )
        for i in range(X.shape[0]):
            X_filtered[i] = mne.filter.filter_data(X[i], **filter_params)
        return X_filtered
# -----------------------------------------------------------------
# --- END OF ADDED BLOCK ---
# -----------------------------------------------------------------

# --- Main Control Function ---
# FIXED (Fix 3): Pass start_time as a parameter
def run_bci_prediction(model_s1, model_s2, meta_s1, meta_s2, raw, start_time):
    """
    Simulates the TWO-STAGE BCI.
    """
    print("\n--- Starting BCI Prediction Loop (Two-Stage) ---")
    print("This will simulate playback in real-time.")
    
    sfreq = meta_s1['sfreq']
    feature_tmin, feature_tmax = meta_s1['feature_window']
    feature_window_samples = int((feature_tmax - feature_tmin) * sfreq)

    # Get class maps from metadata
    class_map_s1 = meta_s1['class_map']
    class_map_s2 = meta_s2['class_map']
    
    
    
    annotations = raw.annotations
    pressed_keys = set() # For robust keyboard control
    
    for onset_s, duration_s, desc in zip(annotations.onset, annotations.duration, annotations.description):
        desc = desc.upper()
        if desc not in ('T0', 'T1', 'T2'):  # ignore irrelevant
            continue

        # FIXED (Fix 4): Correct real-time wait logic
        sim_elapsed_time = time.time() - start_time
        time_to_wait = onset_s - sim_elapsed_time
        if time_to_wait > 0:
            time.sleep(time_to_wait)
            
        final_action = 'IDLE' # Default

        try:
            # 1. Extract the correct data window
            win_start = int((onset_s + feature_tmin) * sfreq)
            win_stop  = int((onset_s + feature_tmax) * sfreq)
            
            data = raw.get_data(picks=meta_s1['channels'], start=win_start, stop=win_stop)
            
            # Sanity check the data shape
            if data.shape[1] != feature_window_samples:
                print(f"Skipping event {desc}: Mismatched data shape. Expected {feature_window_samples}, got {data.shape[1]}")
                continue

            # 2. Reshape for the model
            X = data[np.newaxis, :, :]
            
            # --- STAGE 1 PREDICTION (THE "BRAIN BRAKE") ---
            pred_s1 = model_s1.predict(X)[0]
            # FIXED (Fix 2): Use str(pred_s1) to match JSON keys
            label_s1 = class_map_s1[pred_s1]

 

            if label_s1 == 'intent':
                # --- STAGE 2 PREDICTION (THE "DIRECTION") ---
                pred_s2 = model_s2.predict(X)[0]
                # FIXED (Fix 2): Use str(pred_s2)
                label_s2 = class_map_s2[pred_s2]
                final_action = label_s2.upper()
            else:
                final_action = 'REST'
                
        except Exception as e:
            print(f"Error during prediction at {onset_s:.2f}s: {e}")
            final_action = 'IDLE'

        print(f"[{onset_s:.2f}s] Event: {desc} -> Predicted: {final_action}")
        
        # FIXED (Fix 6): Robust keyboard control
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
    

        print(f"Loading EDF file: {EDF_FILE_PATH}...")
        raw = mne.io.read_raw_edf(EDF_FILE_PATH, preload=True)
        
        print("Applying preprocessing steps to match model training...")
        
        # 1. Normalize channel names
        mne.datasets.eegbci.standardize(raw)
        raw.drop_channels(['T9', 'T10'], on_missing='ignore')
        
        # 2. Apply average reference
        raw.set_eeg_reference('average', projection=False)
        
        # 3. Apply NOTCH filter (bandpass is in the model)
        is_causal = meta_s1.get('causal', False)
        notch_freqs = [50.0] 
        
        if is_causal:
            print(f"Applying CAUSAL Notch filter...")
            raw.notch_filter(freqs=notch_freqs, method='iir', phase='forward')
        else:
            print(f"Applying NON-CAUSAL Notch filter...")
            raw.notch_filter(freqs=notch_freqs, method='fir', fir_design='firwin')
        
        # 4. Pick *only* the channels the model was trained on
        print(f"Selecting {len(meta_s1['channels'])} channels...")
        missing_ch = [ch for ch in meta_s1['channels'] if ch not in raw.ch_names]
        if missing_ch:
            print(f"Error: The EDF file is missing required channels: {missing_ch}")
            exit()
            
        # FIXED (Fix 1): Use pick_channels + reorder_channels
        raw.pick_channels(meta_s1['channels'])
        raw.reorder_channels(meta_s1['channels'])
        
        # FIXED (Fix 5): Add montage
        raw.set_montage('standard_1005')
        
        print("Preprocessing complete.")

        # 3. Run the BCI controller
        print("\nFocus on your game window! Simulation starts in 5 seconds...")
        time.sleep(5)
        start_time = time.time() # Initialize start time
        run_bci_prediction(model_s1, model_s2, meta_s1, meta_s2, raw, start_time)

    except Exception as e:
        print(f"An error occurred: {e}")
