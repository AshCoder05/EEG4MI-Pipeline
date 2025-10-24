import mne
import time
import os
import joblib  # <-- From your training script, to load the model
import numpy as np # <-- Needed for data manipulation
from pynput.keyboard import Key, Controller

# --- Configuration ---

# 1. Paths from your scripts
MODEL_ARTIFACT_PATH = r'D:\mini project\trained_21ch_ovr.joblib'
EDF_FILE_PATH = r'D:\mini project\eeg-motor-movementimagery-dataset-1.0.0\files\S004\S004R07.edf'

# 2. Preprocessing values (MUST match train_group.py)
L_FREQ = 8.0
H_FREQ = 30.0
# The model was trained on data from 1.0s to 4.0s *after* the marker.
FEATURE_TMIN = 1.0
FEATURE_TMAX = 4.0

# 3. Key press mapping
# This map comes from the 'class_map' in your training script's metadata
# {1: 'rest', 2: 'left', 3: 'right'}
CLASS_TO_ACTION = {
    1: 'REST',
    2: 'LEFT',
    3: 'RIGHT'
}

# --- Initialization ---

# Initialize Keyboard Controller (from py_game1.py)
keyboard = Controller()

# --- Main Control Function ---

def run_bci_prediction(model, meta, raw):
    """
    Reads annotations from a pre-processed MNE raw file, extracts the
    EEG data for each trial, predicts the action using the model,
    and sends keyboard commands.
    """
    
    print("\n--- Starting BCI Prediction Loop ---")
    print("This will simulate playback in real-time.")
    print(f"Model expects {len(meta['channels'])} channels: {meta['channels']}")
    print(f"Model feature window: {meta['feature_window']}s")
    
    # Get annotations (from py_game1.py)
    annotations = raw.annotations
    start_time = time.time()
    
    # Iterate through the time-stamped events (from py_game1.py)
    for annot in annotations:
        onset_s = annot['onset']      # Start time of the event
        duration_s = annot['duration'] # Length of the event
        description = annot['description'].upper() # e.g., T0, T1, T2

        # Wait until the simulation reaches the event's start time
        sim_elapsed_time = time.time() - start_time
        time_to_wait = onset_s - sim_elapsed_time
        if time_to_wait > 0:
            time.sleep(time_to_wait)
        
        # --- THIS IS THE NEW, INTEGRATED LOGIC ---
        
        predicted_action = 'IDLE' # Default action
        
        # We only care about the events your model was trained on
        if description == 'T1': # Left fist imagery
            true_event = 'LEFT'
        elif description == 'T2': # Right fist imagery
            true_event = 'RIGHT'
        elif description == 'T0': # Rest
            true_event = 'REST'
            predicted_action = 'REST' # No need to predict for 'rest'
        else:
            true_event = 'OTHER'
            continue # Skip other annotations

        # If it's a motor imagery event, predict from the signal
        if predicted_action != 'REST':
            try:
                # 1. Define the data window based on training script
                t_start = onset_s + FEATURE_TMIN
                t_end = onset_s + FEATURE_TMAX
                
                # 2. Extract the data segment
                # We get the exact time indices from the raw file
                
                # FIX: We rename the variables to '..._arr' to show they are arrays
                start_idx_arr = raw.time_as_index(t_start) 
                stop_idx_arr = raw.time_as_index(t_end)

                # Then, we convert them to simple integers
                start_idx = int(start_idx_arr)
                stop_idx = int(stop_idx_arr)
                
                # Now we pass the simple integers (start_idx) to the function
                data_segment = raw.get_data(picks='eeg', start=start_idx, stop=stop_idx)
                
                # 3. Reshape data for the model
                # The model expects a 3D array: (n_epochs, n_channels, n_times)
                # We have one epoch, so we add a new axis at the start.
                X = data_segment[np.newaxis, :, :] 
                
                # 4. PREDICT!
                pred_label = model.predict(X)[0] # Get the single prediction
                predicted_action = CLASS_TO_ACTION[pred_label]
            
            except Exception as e:
                print(f"Error during prediction at {onset_s:.2f}s: {e}")
                predicted_action = 'IDLE'

        # --- End of NEW Logic ---

        # Print comparison
        print(f"[{onset_s:.2f}s] Event: {description} ({true_event}) -> Model Predicted: {predicted_action}")
        
        # Send keyboard commands (from py_game1.py, but using predicted_action)
        if predicted_action == 'LEFT':
            keyboard.press(Key.left)
            keyboard.release(Key.right)

        elif predicted_action == 'RIGHT':
            keyboard.press(Key.right)
            keyboard.release(Key.left)

        else: # REST or IDLE
            keyboard.release(Key.right)
            keyboard.release(Key.left)
        
        # Wait for the duration of the event (key press held for this long)
        if duration_s > 0:
            time.sleep(duration_s)
    
    # 5. Clean up (from py_game1.py)
    print("\n--- BCI Simulation Finished ---")
    print("Releasing final keys.")
    keyboard.release(Key.right)
    keyboard.release(Key.left)

# --- Main execution ---
if __name__ == "__main__":
    
    # 1. Load the trained model artifact
    if not os.path.exists(MODEL_ARTIFACT_PATH):
        print(f"Error: Model file not found at {MODEL_ARTIFACT_PATH}")
    else:
        print(f"Loading model from {MODEL_ARTIFACT_PATH}...")
        try:
            artifact = joblib.load(MODEL_ARTIFACT_PATH)
            model = artifact['model']
            meta = artifact['meta']
            print("Model and metadata loaded successfully.")
            
            # 2. Load and preprocess the EDF file
            if not os.path.exists(EDF_FILE_PATH):
                print(f"Error: EDF file not found at {EDF_FILE_PATH}")
            else:
                print(f"Loading EDF file: {EDF_FILE_PATH}...")
                raw = mne.io.read_raw_edf(EDF_FILE_PATH, preload=True, stim_channel='auto', verbose='error')
                
                # --- APPLY PREPROCESSING (MUST MATCH train_group.py) ---
                print("Applying preprocessing steps to match model training...")
                
                # Normalize channel names (best-effort)
                try:
                    rename_map = {ch: ch.strip('.').upper() for ch in raw.ch_names}
                    raw.rename_channels(rename_map)
                except Exception:
                    pass
                
                # Apply average reference
                raw.set_eeg_reference('average', projection=False)
                
                # Apply causal bandpass filter
                raw.filter(L_FREQ, H_FREQ, method='iir', iir_params=dict(order=4, ftype='butter'))
                
                # Pick *only* the channels the model was trained on, in the correct order
                missing_ch = [ch for ch in meta['channels'] if ch not in raw.ch_names]
                if missing_ch:
                    print(f"Warning: Missing channels in EDF: {missing_ch}. Model may perform poorly.")
                    # Filter meta channels to only those available
                    channels_to_pick = [ch for ch in meta['channels'] if ch in raw.ch_names]
                else:
                    channels_to_pick = meta['channels']
                
                raw.pick_channels(channels_to_pick, ordered=True)
                
                print("Preprocessing complete.")
                # --- End of Preprocessing ---

                # 3. Run the BCI controller
                print("\nFocus on your game window! Simulation starts in 5 seconds...")
                time.sleep(5)
                run_bci_prediction(model, meta, raw)

        except Exception as e:
            print(f"An error occurred: {e}")