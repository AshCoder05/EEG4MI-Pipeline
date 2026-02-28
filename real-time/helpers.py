import numpy as np
import mne
from scipy.signal import butter, filtfilt
from sklearn.base import BaseEstimator, TransformerMixin


# ==========================================================
# 1. SCIKIT-LEARN COMPATIBLE BANDPASS FILTER CLASS
# (REQUIRED FOR LOADING TRAINED MODEL)
# ==========================================================

class BandpassFilter(BaseEstimator, TransformerMixin):

    def __init__(self, l_freq, h_freq, sfreq, causal=True):
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.sfreq = sfreq
        self.causal = causal

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X_filtered = np.zeros_like(X)

        filter_params = dict(
            sfreq=self.sfreq,
            l_freq=self.l_freq,
            h_freq=self.h_freq,
            method='iir' if self.causal else 'fir',
            iir_params=dict(order=4, ftype='butter', output='sos') if self.causal else None,
            phase='forward' if self.causal else 'zero',
            verbose=False
        )

        if not self.causal:
            filter_params['fir_design'] = 'firwin'

        for i in range(X.shape[0]):
            X_filtered[i] = mne.filter.filter_data(X[i], **filter_params)

        return X_filtered


# ==========================================================
# 2. FUNCTION VERSION (OPTIONAL — FOR OTHER USE)
# ==========================================================

def bandpass_filter(data, sfreq, lowcut=8, highcut=30, order=5):

    nyquist = 0.5 * sfreq
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(order, [low, high], btype='band')

    filtered_data = np.zeros_like(data)

    for trial in range(data.shape[0]):
        for ch in range(data.shape[1]):
            filtered_data[trial, ch, :] = filtfilt(
                b, a, data[trial, ch, :]
            )

    return filtered_data


# ==========================================================
# 3. LOG-VARIANCE TRANSFORM
# ==========================================================

def logvar_transform(data):

    var = np.var(data, axis=2)
    var_norm = var / np.sum(var, axis=1, keepdims=True)
    log_var = np.log(var_norm + 1e-10)

    return log_var