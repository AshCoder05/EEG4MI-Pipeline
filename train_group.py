#train_group.py
"""


Train a single group Left/Right/Rest model on EEGBCI subjects 1-20 using runs:
3,4,7,8,11,12 (left/right + rest). Uses 21 SmartBCI-compatible channels (intersected
with available channels in each recording). Saves model + meta for online use.

Run: python train_bci_21ch_group.py
"""

import json
import joblib
import numpy as np
import mne
from mne.datasets import eegbci
from mne.decoding import CSP
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# --------------------------
# USER CONFIG
# --------------------------
subjects = list(range(1, 31))           # subjects 1..20
selected_runs = [3, 4, 7, 8, 11, 12]   # safe left/right/rest runs
# 21-channel "SmartBCI-compatible" suggestion (will be intersected with actual data)
target_channels = [
    'Fp1','Fp2','AFz','F7','F3','Fz','F4','F8',
    'T7','C3','Cz','C4','T8',
    'P7','P3','Pz','P4','P8',
    'O1','O2','POz'
]
# epoch / feature windows
tmin, tmax = -1.0, 4.0
feature_tmin, feature_tmax = 1.0, 4.0
# filtering
l_freq, h_freq = 8.0, 30.0
causal_filter = True  # set True to simulate causal filtering (iir)
# CSP/classifier
n_csp_components = 6
cv_folds = 5
random_state = 42
# artifact output
model_fname = R"D:\Prsnl\ML\py codes\BCI_Project\src\trained_21ch_ovr.joblib"
meta_fname = R"D:\Prsnl\ML\py codes\BCI_Project\src\trained_21ch_ovr_meta.json"
# rejection
reject_criteria = dict(eeg=150e-6)

# --------------------------
# small helper: log-var
# --------------------------
def logvar_transform(X):
    """X: (n_epochs, n_components, n_times) -> (n_epochs, n_components)"""
    var = X.var(axis=2)
    return np.log(var + 1e-10)

# --------------------------
# 1) collect epochs across subjects
# --------------------------
all_epochs = []
subjects_used = []

print("Starting data collection from EEGBCI (subjects 1..20). This may take some minutes while downloading...")

for subj in subjects:
    try:
        print(f"\n--- Subject {subj:03d} ---")
        files = eegbci.load_data(subj, selected_runs, update_path=True)
        raws = [mne.io.read_raw_edf(f, preload=True, stim_channel='auto') for f in files]
        raw = mne.concatenate_raws(raws)

        # Try to normalize channel names a bit
        try:
            rename_map = {ch: ch.strip('.').upper() for ch in raw.ch_names}
            raw.rename_channels(rename_map)
        except Exception:
            pass

        # montage (best-effort)
        try:
            mont = mne.channels.make_standard_montage('standard_1020')
            raw.set_montage(mont, on_missing='warn')
        except Exception:
            pass

        # apply average reference immediately for reproducibility
        try:
            raw.set_eeg_reference('average', projection=False)
        except Exception:
            pass

        # bandpass filter
        if causal_filter:
            raw.filter(l_freq, h_freq, method='iir', iir_params=dict(order=4, ftype='butter'))
        else:
            raw.filter(l_freq, h_freq, fir_design='firwin')

        # intersect channels
        available = raw.info['ch_names']
        picks = [ch for ch in target_channels if ch in available]
        if len(picks) == 0:
            print("  No target channels found in this recording -> skipping subject.")
            continue
        raw.pick_channels(picks, ordered=False)
        print(f"  Using channels: {picks}")

        # create events from annotations (EEGBCI typically uses T0/T1/T2)
        events, ann_map = mne.events_from_annotations(raw)
        if len(events) == 0:
            print("  No annotations/events -> skipping subject.")
            continue

        # Map T0/T1/T2 semantics for these runs:
        # For runs 3/4/7/8/11/12: T0=rest, T1=left, T2=right (safe)
        # We'll scan annotation descriptions and map only T0/T1/T2 to labels; others ignored.
        kept = []
        for ev in events:
            sample = int(ev[0])
            code = int(ev[2])
            # find description key in ann_map
            desc = None
            for k, v in ann_map.items():
                if v == code:
                    desc = k
                    break
            if desc is None:
                continue
            desc_u = desc.upper()
            if desc_u in ('T0', '0', 'REST'):
                label = 1  # rest
            elif desc_u in ('T1', '1'):
                label = 2  # left
            elif desc_u in ('T2', '2'):
                label = 3  # right
            else:
                # skip unknown annotation descriptions
                continue
            kept.append([sample, 0, label])

        if len(kept) == 0:
            print("  No T0/T1/T2 mapped events -> skip")
            continue

        kept_events = np.array(kept, dtype=int)

        # Build epochs
        event_id = {'rest': 1, 'left': 2, 'right': 3}
        epochs = mne.Epochs(raw, kept_events, event_id=event_id, tmin=tmin, tmax=tmax,
                            picks='eeg', baseline=None, preload=True, reject=reject_criteria)
        epochs.drop_bad()
        if len(epochs) == 0:
            print("  All epochs rejected -> skip")
            continue

        print(f"  Kept {len(epochs)} epochs (after rejection).")
        all_epochs.append(epochs)
        subjects_used.append(subj)

    except Exception as e:
        print(f"  Exception while processing subject {subj}: {e}")
        continue

if len(all_epochs) == 0:
    raise RuntimeError("No epochs collected. Check data availability and channel mapping.")

# --------------------------
# 2) concatenate and prepare X,y
# --------------------------
print(f"\nConcatenating epochs from subjects: {subjects_used}")
group_epochs = mne.concatenate_epochs(all_epochs)
print(f"Total epochs: {len(group_epochs)}. Channels: {len(group_epochs.ch_names)}")

# crop to feature window (exclude cue artifact)
group_epochs_train = group_epochs.copy().crop(tmin=feature_tmin, tmax=feature_tmax)
X = group_epochs_train.get_data()   # shape (n_epochs, n_channels, n_times)
y = group_epochs_train.events[:, -1]
unique, counts = np.unique(y, return_counts=True)
print("Label counts (id -> count):", dict(zip(unique.tolist(), counts.tolist())))

# --------------------------
# 3) build pipeline and cv
# --------------------------
#csp = CSP(n_components=n_csp_components, reg='ledoit_wolf', log=False)
# logvar = FunctionTransformer(logvar_transform, validate=False)
#scaler = StandardScaler()
#lda = LinearDiscriminantAnalysis()

#base_pipe = make_pipeline(CSP(n_components=6), FunctionTransformer(logvar_transform), scaler, lda)
# Let CSP handle log-variance internally and remove our custom function.
csp = CSP(n_components=n_csp_components, reg='ledoit_wolf', log=True)
lda = LinearDiscriminantAnalysis()
base_pipe = make_pipeline(csp, lda) # Simpler pipeline

ovr = OneVsRestClassifier(base_pipe, n_jobs=1)

cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
print("\nRunning within-group cross-validation (stratified)...")
scores = cross_val_score(ovr, X, y, cv=cv, scoring='accuracy', n_jobs=1)
print("CV scores:", scores)
print(f"Mean CV accuracy: {scores.mean()*100:.2f}% (+/- {scores.std()*100:.2f}%)")
print(f"Chance level: {1.0/len(np.unique(y))*100:.2f}%")

# --------------------------
# 4) fit final model and save
# --------------------------
print("\nFitting final model on all data...")
ovr.fit(X, y)

artifact = {
    "model": ovr,
    "meta": {
        "subjects": subjects_used,
        "channels": group_epochs.ch_names,
        "tmin": tmin,
        "tmax": tmax,
        "feature_window": [feature_tmin, feature_tmax],
        "filter": {"l_freq": l_freq, "h_freq": h_freq, "causal": bool(causal_filter)},
        "n_csp_components": n_csp_components,
        "class_map": {"1": "rest", "2": "left", "3": "right"}
    }
}

print(f"Saving model artifact to {model_fname} ...")
joblib.dump(artifact, model_fname)
with open(meta_fname, "w") as f:
    json.dump(artifact["meta"], f, indent=2)

print("Saved model and metadata. Done.")
print("Important: When deploying online, match channel order, reference, filter (use causal filter if online), sliding window size matching feature_window, and use artifact['model'] pipeline to transform/predict.")
