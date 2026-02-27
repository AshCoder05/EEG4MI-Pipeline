import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Keep warnings visible
mne.set_log_level('WARNING')

# ==========================================
# 1. CONFIGURATION
# ==========================================
TEST_FILE = R"D:\Prsnl\ML\py codes\BCI_Project\src\demo\alldata\S011_001_EEG.edf"

full_19_channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2']

channel_map = {
    'EEG Fp1': 'Fp1', 'EEG Fp1-Ref': 'Fp1', 'EEG Fp2': 'Fp2', 'EEG Fp2-Ref': 'Fp2',
    'EEG F7': 'F7', 'EEG F7-Ref': 'F7', 'EEG F3': 'F3', 'EEG F3-Ref': 'F3', 
    'EEG Fz': 'Fz', 'EEG Fz-Ref': 'Fz', 'EEG F4': 'F4', 'EEG F4-Ref': 'F4', 
    'EEG T3': 'T7', 'EEG T3-Ref': 'T7', 'EEG C3': 'C3', 'EEG C3-Ref': 'C3', 
    'EEG Cz': 'Cz', 'EEG Cz-Ref': 'Cz', 'EEG C4': 'C4', 'EEG C4-Ref': 'C4', 
    'EEG T4': 'T8', 'EEG T4-Ref': 'T8', 'EEG T5': 'P7', 'EEG T5-Ref': 'P7', 
    'EEG P3': 'P3', 'EEG P3-Ref': 'P3', 'EEG Pz': 'Pz', 'EEG Pz-Ref': 'Pz', 
    'EEG P4': 'P4', 'EEG P4-Ref': 'P4', 'EEG T6': 'P8', 'EEG T6-Ref': 'P8', 
    'EEG O1': 'O1', 'EEG O1-Ref': 'O1', 'EEG O2': 'O2', 'EEG O2-Ref': 'O2'
}
# Reject any epoch where the peak-to-peak voltage exceeds 150 microvolts
reject_criteria = dict(eeg=150e-6)
# ==========================================
# 2. LOAD & PREPROCESS (Targeting Beta Band)
# ==========================================
print(f"Loading {os.path.basename(TEST_FILE)}...")
raw = mne.io.read_raw_edf(TEST_FILE, preload=True)
raw.rename_channels({k: v for k, v in channel_map.items() if k in raw.ch_names})
raw.set_montage('standard_1005', on_missing='ignore')
raw.pick_channels([ch for ch in full_19_channels if ch in raw.ch_names])

# Clean the baseline and target the 16-24Hz Beta band directly
raw.filter(l_freq=16.0, h_freq=24.0, phase='zero')
raw.notch_filter(freqs=[50.0], phase='zero')

# ==========================================
# 3. EPOCHING
# ==========================================
def direction_mapper(description):
    desc_lower = str(description).lower()
    if 'left' in desc_lower: return 0
    if 'right' in desc_lower: return 1
    return None

events, _ = mne.events_from_annotations(raw, event_id=direction_mapper)
epochs = mne.Epochs(raw, events, event_id={'Left': 0, 'Right': 1}, tmin=-0.5, tmax=3.0, baseline=(-0.5, 0.0), preload=True,reject=reject_criteria)
epochs.crop(tmin=0.5, tmax=2.5)

X = epochs.get_data()
y = epochs.events[:, -1]

# ==========================================
# 4. FIT CSP & EXTRACT FEATURES
# ==========================================
print("\nFitting CSP...")
# We use exactly 2 components to create a perfect 2D X/Y scatter plot
csp = CSP(n_components=2, reg='ledoit_wolf', log=True)
X_features = csp.fit_transform(X, y)

print("Plotting CSP Patterns (Biology) vs Filters (Math)...")
fig_pat = csp.plot_patterns(epochs.info, ch_type='eeg', show=False)
fig_pat.suptitle("CSP PATTERNS (Where the brain is firing)", fontsize=14)

fig_filt = csp.plot_filters(epochs.info, ch_type='eeg', show=False)
fig_filt.suptitle("CSP FILTERS (How the math cancels noise)", fontsize=14)
plt.show()

# ==========================================
# 5. FIT LDA & PLOT DECISION BOUNDARY
# ==========================================
print("\nFitting LDA...")
lda = LinearDiscriminantAnalysis(solver='svd')
lda.fit(X_features, y)

# Create a meshgrid to plot the mathematical decision boundary
x_min, x_max = X_features[:, 0].min() - 1, X_features[:, 0].max() + 1
y_min, y_max = X_features[:, 1].min() - 1, X_features[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

# Predict across the entire grid
Z = lda.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
# Plot the LDA boundary
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
# Plot the actual feature points
plt.scatter(X_features[y == 0, 0], X_features[y == 0, 1], color='blue', edgecolors='k', label='Left Intent', s=70)
plt.scatter(X_features[y == 1, 0], X_features[y == 1, 1], color='red', edgecolors='k', label='Right Intent', s=70)

plt.title("LDA Decision Boundary in CSP Feature Space")
plt.xlabel("CSP Component 1 (Log-Variance)")
plt.ylabel("CSP Component 2 (Log-Variance)")
plt.legend()
plt.show()

# ==========================================
# 6. LDA 1D PROJECTION (HISTOGRAMS)
# ==========================================
print("\nPlotting LDA Projection Histograms...")
# Transform data to the 1D axis that LDA calculates maximizes class distance
X_lda_1d = lda.transform(X_features)

plt.figure(figsize=(10, 5))
plt.hist(X_lda_1d[y == 0], bins=15, alpha=0.6, color='blue', label='Left Intent', edgecolor='black')
plt.hist(X_lda_1d[y == 1], bins=15, alpha=0.6, color='red', label='Right Intent', edgecolor='black')

plt.title("LDA 1D Projection (Class Separation)")
plt.xlabel("Distance from Decision Boundary (0 = Boundary)")
plt.ylabel("Epoch Count")
plt.legend()
plt.axvline(0, color='black', linestyle='--', linewidth=2, label="Decision Boundary")
plt.show()

print("\n✅ CSP/LDA Visualization Complete.")

# ==========================================
# DIAGNOSTIC: WHY IS IT DIAGONAL?
# ==========================================
print("\n--- FORENSIC MATH CHECK ---")
# 1. Check the data scale (Did the hardware scale the EDF wrong?)
print(f"Max Voltage in Data: {np.max(X):.5f} Volts")

# 2. Check the Rank (Did the channels bridge and become identical?)
print(f"Spatial Matrix Rank: {np.linalg.matrix_rank(X[0])} (Should be 19)")

# 3. Check the Eigenvalues (Are the classes actually different?)
# A good CSP has eigenvalues near 1.0 and 0.0. 
# If they are exactly 0.5, the classes are identical to the algorithm.
evals = csp.eigenvalue_
print(f"CSP Eigenvalues: {evals}")
print("---------------------------\n")