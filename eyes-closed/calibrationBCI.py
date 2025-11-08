#!/usr/bin/env python3
"""
calibrate_s6_FINETUNE.py

This script implements the "Gradual User Enrollment" methodology
(Transfer Learning / Fine-Tuning) for Subject 's6'.

Strategy:
- We use the (0.0, 2.0)s time window. THis feature window needs to be changed as needed for each subject.
- We use the "Fine-Tune" (Frozen FBCSP) strategy for BOTH stages,
  as this was proven most stable for s6's small dataset.
"""

import joblib
import numpy as np
import mne
import os
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# --- CRITICAL: Imports needed for joblib to load the pipeline ---
from mne.decoding import CSP
from sklearn.preprocessing import FunctionTransformer, StandardScaler

# --- NEW IMPORTS FOR VALIDATION ---
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    confusion_matrix, 
    ConfusionMatrixDisplay
)

# Set MNE to be less "chatty"
mne.set_log_level('WARNING')

# --------------------------
# USER CONFIG
# --------------------------

# --- 1. Paths to YOUR new 's6' data ---
# !!! UPDATE THESE PATHS !!!
S6_DEDICATED_REST_FILES = []
S6_MI_FILES = [
    R"D:\Prsnl\ML\py codes\BCI_Project\src\yaboi\S06_s001_EEG.edf",
    # Add other s6 MI files if you have them
    # R"D:\path\to\s6\S06_s002_EEG.edf", 
]

# --- 2. Paths to the *GENERAL 19-CHANNEL* models you just trained ---
GENERAL_MODEL_S1_FNAME = R"D:\Prsnl\ML\py codes\BCI_Project\src\bci_stage1_rest_intent_model_19ch.joblib"
GENERAL_MODEL_S2_FNAME = R"D:\Prsnl\ML\py codes\BCI_Project\src\bci_stage2_left_right_model_19ch.joblib"

# --- 3. Paths for the *NEW, CALIBRATED* models to be saved ---
CALIBRATED_MODEL_S1_FNAME = R"D:\Prsnl\ML\py codes\BCI_Project\src\s6_calibrated_stage1_model.joblib"
CALIBRATED_MODEL_S2_FNAME = R"D:\Prsnl\ML\py codes\BCI_Project\src\s6_calibrated_stage2_model.joblib"


# --- 4. CONFIG (Must match the general model's training) ---
model_channels = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 
    'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2'
] 
tmin, tmax = -0.5, 3.0 # Epoch window

# --- *** BEST TIME WINDOW based on your tests *** ---
feature_tmin, feature_tmax = 0, 2

epoch_duration_for_rest = 3.5 # tmax - tmin
causal_filter = False
notch_freqs = [50.0]
global_sfreq = 160.0
MI_FILE_REST_DURATION_SEC = 180.0
reject_criteria = None


# -------------------------------------------------------------
# --- CRITICAL BLUEPRINTS ---
# -------------------------------------------------------------

def logvar_transform(X):
    var = X.var(axis=2)
    return np.log(var + 1e-10)

class BandpassFilter(BaseEstimator, TransformerMixin):
    def __init__(self, l_freq, h_freq, sfreq, causal=False):
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.sfreq = sfreq
        self.causal = causal
    def fit(self, X, y=None): return self
    def transform(self, X):
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

# -------------------------------------------------------------
# --- DATA LOADING FUNCTION ---
# -------------------------------------------------------------

def load_and_preprocess_s6_raw(file_list):
    """
    Loads, renames 's6' data to be compatible with PhysioNet,
    and preprocesses it.
    """
    if not file_list:
        print("Warning: No files provided to load_and_preprocess_s6_raw.")
        return None
        
    try:
        raws = [mne.io.read_raw_edf(f, preload=True, verbose=False) for f in file_list]
    except FileNotFoundError as e:
        print(f"FATAL ERROR: File not found. Did you update the path?")
        print(f"{e}")
        return None
        
    raw = mne.concatenate_raws(raws)
    
    # --- *** CRITICAL: VERIFY THIS MAP FOR 's6' *** ---
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
        print(f"--- WARNING: Channel renaming failed. ---")
        print("This is OK if your 's6' channels already match the model (e.g., 'Fp1', 'C3', 'T7').")
        print(f"Details: {e}")

    raw.drop_channels(['A1', 'A2'], on_missing='ignore')

    available = raw.info['ch_names']
    picks_model = [ch for ch in model_channels if ch in available]
    if len(picks_model) < 19:
        missing = set(model_channels) - set(available)
        raise RuntimeError(f"Could not find all 19 model channels after renaming. Missing: {missing}")
    raw.pick_channels(picks_model, ordered=False)
    
    raw.set_montage('standard_1005', on_missing='ignore')
    raw.set_eeg_reference('average', projection=False)
    raw.notch_filter(freqs=notch_freqs, method='fir', fir_design='firwin')
    
    if raw.info['sfreq'] != global_sfreq:
        print(f"Resampling data from {raw.info['sfreq']}Hz to {global_sfreq}Hz...")
        raw.resample(global_sfreq)
        
    return raw

# -------------------------------------------------------------
# --- MODELING FUNCTIONS ---
# -------------------------------------------------------------

def _get_model_from_file(general_model_path):
    """Helper function to load the pipeline from a file (that might be a dict)."""
    loaded_object = joblib.load(general_model_path)
    
    if isinstance(loaded_object, dict):
        print("Loaded model data as a dictionary. Extracting 'model' pipeline...")
        if 'model' not in loaded_object:
            print("ERROR: Loaded dictionary does not have a 'model' key.")
            return None
        model = loaded_object['model']
    elif hasattr(loaded_object, 'steps'):
        print("Loaded model data as a direct pipeline.")
        model = loaded_object
    else:
        print(f"ERROR: Loaded object is not a pipeline or a known dictionary.")
        return None
    
    return model

def evaluate_calibration(general_model_path, X_calib, y_calib, class_names):
    """
    Performs cross-validation on the calibration data to get performance scores
    and a confusion matrix *before* final training.
    
    This function ALWAYS uses the "Fine-Tune" (Frozen FBCSP) strategy.
    """
    print("\n" + "-"*30)
    print(f" VALIDATING CALIBRATION: {class_names[0]} vs {class_names[1]}")
    print(f" Using {len(y_calib)} total trials...")
    print(f" Loading general model from {general_model_path}...")
    
    model = _get_model_from_file(general_model_path)
    if model is None:
        raise RuntimeError("Failed to load model for evaluation.")

    # 3. Define our cross-validation strategy
    n_splits = 5
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    print(f"Running 5-Fold Cross-Validation...")
    print("Validation strategy: FINE-TUNE (Frozen FBCSP, retrain LDA)")

    # 1. Extract features using the "frozen" extractor
    print("Extracting features from calibration data...")
    feature_extractor = Pipeline(model.steps[:-1])
    X_features = feature_extractor.transform(X_calib)

    # 2. Get a *clone* of the final classifier.
    classifier = clone(model.steps[-1][1])
    y_pred_cv = cross_val_predict(classifier, X_features, y_calib, cv=cv)
    y_for_matrix = y_calib


    # 5. Calculate metrics
    labels_list = [1, 2] 
    accuracy = accuracy_score(y_for_matrix, y_pred_cv)
    precision_per_class = precision_score(y_for_matrix, y_pred_cv, average=None, labels=labels_list, zero_division=0)
    recall_per_class = recall_score(y_for_matrix, y_pred_cv, average=None, labels=labels_list, zero_division=0)

    # 6. Print the results
    print("\n--- VALIDATION SCORES (from 5-Fold CV) ---")
    print(f"  Overall Accuracy:  {accuracy:.3f}")
    print("\n  Per-Class Scores:")
    for i, class_name in enumerate(class_names):
        print(f"    --- {class_name} (Label {labels_list[i]}) ---")
        print(f"    Precision: {precision_per_class[i]:.3f}")
        print(f"    Recall:    {recall_per_class[i]:.3f}")
    
    # 7. Generate and plot the confusion matrix
    print("\nGenerating Cross-Validated Confusion Matrix...")
    cm = confusion_matrix(y_for_matrix, y_pred_cv, labels=labels_list)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    disp.ax_.set_title(f"'{class_names[0]}' vs '{class_names[1]}' (CV Matrix)")
    plt.show()
    
    print("--- VALIDATION COMPLETE ---")


def calibrate_model(general_model_path, X_calib, y_calib):
    """
    Loads a general model and fine-tunes its classifier layer.
    (Freezes FBCSP, retrains LDA)
    """
    print(f"Loading general model from {general_model_path} for FINE-TUNE training...")
    
    model = _get_model_from_file(general_model_path)
    if model is None:
        raise RuntimeError("Failed to load model for calibration.")

    feature_extractor = Pipeline(model.steps[:-1])
    classifier = model.steps[-1][1] 

    print(f"Extracting features from {len(X_calib)} calibration trials...")
    X_calib_features = feature_extractor.transform(X_calib)

    print(f"Fine-tuning {type(classifier).__name__} on ALL calibration data...")
    classifier.fit(X_calib_features, y_calib)
    
    print("Final fine-tuning complete.")
    return model


# --------------------------
# --- MAIN CALIBRATION SCRIPT ---
# --------------------------
if __name__ == "__main__":
    try:
        # --- STAGE 1: CALIBRATE "REST" vs. "INTENT" ---
        print("="*60)
        print(" STAGE 1: CALIBRATING 'REST' vs. 'INTENT' for s6")
        print("="*60)
        
        # --- S6 DATA LOADING ---
        print("Loading S6 data...")
        raw_rest_dedicated = load_and_preprocess_s6_raw(S6_DEDICATED_REST_FILES)
        raw_mi_full = load_and_preprocess_s6_raw(S6_MI_FILES)
        
        print(f"Creating 'rest' data from the first {MI_FILE_REST_DURATION_SEC}s of MI files...")
        raw_rest_from_mi = raw_mi_full.copy().crop(tmin=0.0, tmax=MI_FILE_REST_DURATION_SEC, include_tmax=False)
        
        print(f"Creating 'intent' data from {MI_FILE_REST_DURATION_SEC}s onwards...")
        raw_intent = raw_mi_full.copy().crop(tmin=MI_FILE_REST_DURATION_SEC)
        
        # --- ANNOTATION CHECK ---
        print("--- ANNOTATION CHECK ---")
        print(np.unique(raw_intent.annotations.description))
        # ---

        print("Combining dedicated rest and MI-rest data...")
        if raw_rest_dedicated:
            raw_rest = mne.concatenate_raws([raw_rest_dedicated, raw_rest_from_mi])
        else:
            print("No dedicated rest files found, using only MI-rest.")
            raw_rest = raw_rest_from_mi

        if raw_rest is None or raw_intent is None:
             raise FileNotFoundError("Could not create valid rest or intent data.")
        
        # --- EPOCHING S1 ---
        event_id_s1 = {'REST': 1, 'INTENT': 2}
        
        events_rest = mne.make_fixed_length_events(raw_rest, id=event_id_s1['REST'], duration=epoch_duration_for_rest)
        epochs_rest = mne.Epochs(raw_rest, events_rest, event_id=event_id_s1, 
                                 tmin=tmin, tmax=tmax, picks='eeg', baseline=None, 
                                 preload=True, reject=reject_criteria, on_missing='warn') 
        
        def intent_mapper(description):
            desc_lower = str(description).lower()
            if 'left' in desc_lower or 'right' in desc_lower:
                return event_id_s1['INTENT']
            else:
                return None
        
        events_intent, _ = mne.events_from_annotations(raw_intent, event_id=intent_mapper)
        epochs_intent = mne.Epochs(raw_intent, events_intent, event_id=event_id_s1, 
                                   tmin=tmin, tmax=tmax, picks='eeg', baseline=None, 
                                   preload=True, reject=reject_criteria, on_missing='warn')
        
        if 'INTENT' not in epochs_intent.event_id:
             raise RuntimeError("Could not find any 'left' or 'right' markers in the MI files.")
        epochs_intent = epochs_intent['INTENT']

        epochs_s1 = mne.concatenate_epochs([epochs_rest, epochs_intent])
        epochs_s1.equalize_event_counts(['REST', 'INTENT']) 
        epochs_s1.crop(tmin=feature_tmin, tmax=feature_tmax)
        
        X_s1 = epochs_s1.get_data()
        y_s1 = epochs_s1.events[:, -1]
        
        print(f"Created Stage 1 calibration set: {X_s1.shape[0]} trials")
        
        # --- S1: VALIDATE (Fine-tune strategy) ---
        evaluate_calibration(GENERAL_MODEL_S1_FNAME, X_s1, y_s1, class_names=['REST', 'INTENT'])

        # --- S1: TRAIN & SAVE (Fine-tune strategy) ---
        calibrated_model_s1 = calibrate_model(GENERAL_MODEL_S1_FNAME, X_s1, y_s1)
        if calibrated_model_s1:
            joblib.dump(calibrated_model_s1, CALIBRATED_MODEL_S1_FNAME)
            print(f"✅ Stage 1 calibrated model saved to {CALIBRATED_MODEL_S1_FNAME}")
        else:
            raise RuntimeError("Stage 1 calibration failed.")


        # --- STAGE 2: CALIBRATE "LEFT" vs. "RIGHT" ---
        print("\n" + "="*60)
        print(" STAGE 2: CALIBRATING 'LEFT' vs. 'RIGHT' for s6")
        print("="*60)

        # --- EPOCHING S2 ---
        event_id_s2 = {'LEFT': 1, 'RIGHT': 2}

        def direction_mapper(description):
            desc_lower = str(description).lower()
            if 'left' in desc_lower:
                return event_id_s2['LEFT']
            elif 'right' in desc_lower:
                return event_id_s2['RIGHT']
            else:
                return None
                
        events_s2, _ = mne.events_from_annotations(raw_intent, event_id=direction_mapper)
        
        epochs_s2 = mne.Epochs(raw_intent, events_s2, event_id=event_id_s2, 
                               tmin=tmin, tmax=tmax, picks='eeg', baseline=None, 
                               preload=True, reject=reject_criteria, on_missing='warn')

        if 'LEFT' not in epochs_s2.event_id or 'RIGHT' not in epochs_s2.event_id:
            raise RuntimeError("Could not find both 'left' and 'right' markers in the MI files.")
        epochs_s2 = epochs_s2['LEFT', 'RIGHT']

        epochs_s2.equalize_event_counts(['LEFT', 'RIGHT'])
        epochs_s2.crop(tmin=feature_tmin, tmax=feature_tmax)
        
        X_s2 = epochs_s2.get_data()
        y_s2 = epochs_s2.events[:, -1]

        print(f"Created Stage 2 calibration set: {X_s2.shape[0]} trials")
        
        # --- S2: VALIDATE (Fine-tune strategy) ---
        # We are now using the "Fine-Tune" strategy for Stage 2
        evaluate_calibration(GENERAL_MODEL_S2_FNAME, X_s2, y_s2, class_names=['LEFT', 'RIGHT'])

        # --- S2: TRAIN & SAVE (Fine-tune strategy) ---
        # We are now using the "Fine-Tune" strategy for Stage 2
        calibrated_model_s2 = calibrate_model(GENERAL_MODEL_S2_FNAME, X_s2, y_s2)
        if calibrated_model_s2:
            joblib.dump(calibrated_model_s2, CALIBRATED_MODEL_S2_FNAME)
            print(f"✅ Stage 2 calibrated model saved to {CALIBRATED_MODEL_S2_FNAME}")
        else:
            raise RuntimeError("Stage 2 calibration failed.")
        
        print("\n--- 's6' Calibration Complete ---")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
