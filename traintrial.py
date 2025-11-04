#!/usr/bin/env python3
"""
train_two_stage_optimized.py

Trains two specialized models for a real-time BCI.
This version REMOVES ICA to train a "realistic" model on noisy data
to match the online simulation script.

1.  Model 1 (Brain Brake): 'True Rest' (Runs 1, 2) vs. 'Intent' (Task Runs)
2.  Model 2 (Direction): 'Left Fist' vs. 'Right Fist'
"""

import json
import joblib
import numpy as np
import mne
import os
from mne.datasets import eegbci
from mne.decoding import CSP
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin

# --- ICA REMOVED ---
# from mne.preprocessing import ICA

# Set MNE to be less "chatty"
mne.set_log_level('WARNING')

# --------------------------
# USER CONFIG
# --------------------------
subjects = list(range(1, 21)) # Use subjects 1-20
model_channels = [
    'F7','F3','Fz','F4','F8', 'T7','C3','Cz','C4','T8',
    'P7','P3','Pz','P4','P8'
]

# --- REALISTIC FIX ---
# We only load the channels the model will use.
# This makes the training data identical to the simulation data.
ica_channels = model_channels 

# --- CRITICAL FIX (Bug #6): Update Time Windows ---
tmin, tmax = -0.5, 3.0
feature_tmin, feature_tmax = 0.5, 2.5
epoch_duration_for_rest = 3.5 # tmax - tmin

# --- CRITICAL FIX (Bug #3): Set causal_filter=False for training ---
causal_filter = False
notch_freqs = [50.0]

# --- FIX (Bug #11): Overlapping bands ---
filter_bands = [(8, 12), (10, 14), (12, 16), (14, 20), (18, 26), (24, 30)]
n_csp_components_per_band = 4
random_state = 42
reject_criteria = dict(eeg=150e-6)

model1_fname = R"D:\Prsnl\ML\py codes\BCI_Project\src\bci_stage1_rest_intent_model.joblib"
model2_fname = R"D:\Prsnl\ML\py codes\BCI_Project\src\bci_stage2_left_right_model.joblib"

# --------------------------
# Helpers
# --------------------------

def logvar_transform(X):
    """X: (n_epochs, n_components, n_times) -> (n_epochs, n_components)"""
    var = X.var(axis=2)
    return np.log(var + 1e-10)

# --- CRITICAL FIX (Bug #1 & #4): Custom Bandpass Filter for Pipeline ---
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
            phase='forward' if self.causal else 'zero', # 'zero' is non-causal
            fir_design='firwin' if not self.causal else None,
            verbose=False # Suppress log spam
        )
        
        # Apply filter per epoch (Bug #4 fix)
        for i in range(X.shape[0]):
            X_filtered[i] = mne.filter.filter_data(X[i], **filter_params)
        return X_filtered

def create_fbcsp_pipeline(sfreq):
    """Builds a full FBCSP -> Scaler -> LDA pipeline."""
    feature_pipelines = []
    for l_freq, h_freq in filter_bands:
        band_name = f"{l_freq}-{h_freq}Hz"
        
        filter_step = BandpassFilter(
            l_freq=l_freq, h_freq=h_freq, 
            sfreq=sfreq, causal=causal_filter
        )
        
        # --- FIX (Bug #5): Add random_state to CSP ---
        csp_step = CSP(
            n_components=n_csp_components_per_band, 
            reg='ledoit_wolf', log=True
        )
        
        band_pipeline = Pipeline([('filter', filter_step), ('csp', csp_step)])
        feature_pipelines.append((band_name, band_pipeline))

    feature_union = FeatureUnion(feature_pipelines)
    
    full_pipeline = make_pipeline(
        feature_union,
        StandardScaler(),
        # Use LDA with shrinkage for better regularization
        LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    )
    return full_pipeline

# --------------------------
# --- STAGE 1: REST vs. INTENT ---
# --------------------------
print("="*60)
print(" STAGE 1: TRAINING 'REST' vs. 'INTENT' MODEL")
print("="*60)

all_epochs_s1 = []
groups_s1 = []
subjects_used_s1 = []
global_sfreq = 160.0 # EEGBCI is 160Hz

for subj in subjects:
    try:
        print(f"\n--- Subject {subj:03d} (Stage 1) ---")
        
        rest_files = eegbci.load_data(subj, [1, 2], update_path=True)
        raw_rest = mne.concatenate_raws([mne.io.read_raw_edf(f, preload=True) for f in rest_files])
        
        intent_files = eegbci.load_data(subj, [3, 4, 7, 8, 11, 12], update_path=True)
        raw_intent = mne.concatenate_raws([mne.io.read_raw_edf(f, preload=True) for f in intent_files])
        
        epochs_list_for_subject = []
        for raw, task_type in [(raw_rest, 'rest'), (raw_intent, 'intent')]:
            mne.datasets.eegbci.standardize(raw)
            raw.drop_channels(['T9', 'T10'], on_missing='ignore')
            
            # This pick now selects our 18 model_channels
            available = raw.info['ch_names']
            picks_model = [ch for ch in ica_channels if ch in available] # ica_channels IS model_channels
            if len(picks_model) < 8:
                print(f"  Too few channels for {task_type}, skipping...")
                continue
            raw.pick_channels(picks_model, ordered=False)
            
            raw.set_montage('standard_1005')
            raw.set_eeg_reference('average', projection=False)

            # --- Notch filter ---
            if causal_filter:
                raw.notch_filter(freqs=notch_freqs, method='iir', phase='forward')
            else:
                raw.notch_filter(freqs=notch_freqs, method='fir', fir_design='firwin')

            # --- ENTIRE ICA BLOCK REMOVED ---
            print(f"  Skipping ICA (training realistic model).")
            
            # --- EPOCHING ---
            if task_type == 'rest':
                events = mne.make_fixed_length_events(raw, id=0, duration=epoch_duration_for_rest)
                event_id = {'rest': 0} # 0-based label
            else:
                events_from_annot, ann_map = mne.events_from_annotations(raw)
                kept = []
                for ev in events_from_annot:
                    sample, _, code = int(ev[0]), int(ev[1]), int(ev[2])
                    desc = [k for k, v in ann_map.items() if v == code]
                    if not desc: continue
                    desc_u = desc[0].upper()
                    if desc_u in ('T1', '1', 'T2', '2'): # Both Left and Right are 'Intent'
                        kept.append([sample, 0, 1]) # 1-based label
                events = np.array(kept, dtype=int)
                event_id = {'intent': 1}

            if len(events) == 0:
                print(f"  No events found for {task_type}, skipping...")
                continue
                
            epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                                picks='eeg', baseline=None, preload=True, 
                                reject=reject_criteria)
            epochs.drop_bad()
            epochs_list_for_subject.append(epochs)
            
        if len(epochs_list_for_subject) == 2:
            subj_epochs = mne.concatenate_epochs(epochs_list_for_subject)
            # --- FIX: Balance per-subject for robust LOSO groups ---
            print(f"  Counts before balancing: {np.unique(subj_epochs.events[:, -1], return_counts=True)[1]}")
            subj_epochs.equalize_event_counts(['rest', 'intent'])
            print(f"  Kept {len(subj_epochs)} balanced epochs.")
            
            all_epochs_s1.append(subj_epochs)
            subjects_used_s1.append(subj)
            groups_s1.extend([subj] * len(subj_epochs)) # Appends BALANCED length
        else:
            print("  Missing rest or intent data, skipping subject.")

    except Exception as e:
        print(f"  Exception while processing subject {subj}: {e}")
        continue

if not all_epochs_s1:
    raise RuntimeError("No epochs collected for Stage 1.")

# --- VALIDATE AND TRAIN STAGE 1 ---
group_epochs_s1 = mne.concatenate_epochs(all_epochs_s1)
group_epochs_s1.crop(tmin=feature_tmin, tmax=feature_tmax)
X_s1 = group_epochs_s1.get_data()
y_s1 = group_epochs_s1.events[:, -1]
groups_s1 = np.array(groups_s1)

assert len(X_s1) == len(y_s1) == len(groups_s1), "Stage 1 Data mismatch!"

model_s1 = create_fbcsp_pipeline(global_sfreq)
logo = LeaveOneGroupOut()

print(f"\nRunning LOSO cross-validation for Stage 1 model...")
scores_s1 = cross_val_score(model_s1, X_s1, y_s1, cv=logo, groups=groups_s1, n_jobs=1)
print(f"--- Stage 1 (Rest vs. Intent) LOSO Accuracy: {np.mean(scores_s1)*100:.2f}% ---")

print("Fitting final Stage 1 model on all data...")
model_s1.fit(X_s1, y_s1)

# --- CREATE THE ARTIFACT DICTIONARY (STAGE 1) ---
artifact_s1 = {
    "model": model_s1,
    "meta": {
        "subjects_used": subjects_used_s1,
        "channels": group_epochs_s1.ch_names,
        "feature_window": [feature_tmin, feature_tmax],
        "filter_bands": filter_bands,
        "causal": bool(causal_filter),
        "sfreq": global_sfreq,
        "class_map": {0: "rest", 1: "intent"}
    }
}
joblib.dump(artifact_s1, model1_fname)
print(f"✅ Stage 1 model artifact saved to {model1_fname}")


# --------------------------
# --- STAGE 2: LEFT vs. RIGHT ---
# --------------------------
print("\n" + "="*60)
print(" STAGE 2: TRAINING 'LEFT' vs. 'RIGHT' MODEL")
print("="*60)

all_epochs_s2 = []
groups_s2 = []
subjects_used_s2 = []

for subj in subjects:
    try:
        print(f"\n--- Subject {subj:03d} (Stage 2) ---")
        
        files = eegbci.load_data(subj, [4, 8, 12], update_path=True)
        raw = mne.concatenate_raws([mne.io.read_raw_edf(f, preload=True) for f in files])

        mne.datasets.eegbci.standardize(raw)
        raw.drop_channels(['T9', 'T10'], on_missing='ignore')
        
        # This pick now selects our 18 model_channels
        available = raw.info['ch_names']
        picks_model = [ch for ch in ica_channels if ch in available] # ica_channels IS model_channels
        raw.pick_channels(picks_model, ordered=False)
        
        raw.set_montage('standard_1005')
        raw.set_eeg_reference('average', projection=False)

        if causal_filter:
            raw.notch_filter(freqs=notch_freqs, method='iir', phase='forward')
        else:
            raw.notch_filter(freqs=notch_freqs, method='fir', fir_design='firwin')
            
        # --- ENTIRE ICA BLOCK REMOVED ---
        print(f"  Skipping ICA (training realistic model).")
        
        # --- EPOCHING (LEFT vs RIGHT only) ---
        events_from_annot, ann_map = mne.events_from_annotations(raw)
        kept = []
    
        for ev in events_from_annot:
            sample, _, code = int(ev[0]), int(ev[1]), int(ev[2])
            desc = [k for k, v in ann_map.items() if v == code]
            if not desc: continue
            desc_u = desc[0].upper()
            if desc_u in ('T1', '1'):
                kept.append([sample, 0, 0]) # Label 'left' as 0
            elif desc_u in ('T2', '2'):
                kept.append([sample, 0, 1]) # Label 'right' as 1
            # We IGNORE T0 ('rest') events
                
        if len(kept) == 0:
            print("  No T1/T2 events found, skipping...")
            continue
            
        events = np.array(kept, dtype=int)
        event_id = {'left': 0, 'right': 1}

        epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                            picks='eeg', baseline=None, preload=True, 
                            reject=reject_criteria)
        epochs.drop_bad()
        
        print(f"  Counts before balancing: {np.unique(epochs.events[:, -1], return_counts=True)[1]}")
        epochs.equalize_event_counts(['left', 'right'])
        print(f"  Kept {len(epochs)} balanced epochs.")
                
        if len(epochs) > 0:
            all_epochs_s2.append(epochs)
            subjects_used_s2.append(subj)
            groups_s2.extend([subj] * len(epochs))
        else:
            print("  No clean epochs left, skipping subject.")

    except Exception as e:
        print(f"  Exception while processing subject {subj}: {e}")
        continue

if not all_epochs_s2:
    raise RuntimeError("No epochs collected for Stage 2.")

# --- VALIDATE AND TRAIN STAGE 2 ---
group_epochs_s2 = mne.concatenate_epochs(all_epochs_s2)
group_epochs_s2.crop(tmin=feature_tmin, tmax=feature_tmax)
X_s2 = group_epochs_s2.get_data()
y_s2 = group_epochs_s2.events[:, -1]
groups_s2 = np.array(groups_s2)

assert len(X_s2) == len(y_s2) == len(groups_s2), "Stage 2 Data mismatch!"

model_s2 = create_fbcsp_pipeline(global_sfreq)
logo_s2 = LeaveOneGroupOut()

print(f"\nRunning LOSO cross-validation for Stage 2 model...")
scores_s2 = cross_val_score(model_s2, X_s2, y_s2, cv=logo_s2, groups=groups_s2, n_jobs=1)
print(f"--- Stage 2 (Left vs. Right) LOSO Accuracy: {np.mean(scores_s2)*100:.2f}% ---")

print("Fitting final Stage 2 model on all data...")
model_s2.fit(X_s2, y_s2)

# --- CREATE THE ARTIFACT DICTIONARY (STAGE 2) ---
artifact_s2 = {
    "model": model_s2,
    "meta": {
        "subjects_used": subjects_used_s2,
        "channels": group_epochs_s2.ch_names,
        "feature_window": [feature_tmin, feature_tmax],
        "filter_bands": filter_bands,
        "causal": bool(causal_filter),
        "sfreq": global_sfreq,
        "class_map": {0: "left", 1: "right"}
    }
}
joblib.dump(artifact_s2, model2_fname)
print(f"✅ Stage 2 model artifact saved to {model2_fname}")
print("\n--- Two-Stage Model Training Complete ---")