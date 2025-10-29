import numpy as np
from pylsl import StreamInlet, resolve_stream
from scipy.signal import welch, butter, lfilter

# --- SETTINGS ---
SAMPLING_RATE = 250  # IMPORTANT: Change this to your headset's actual sampling rate
EPOCH_LENGTH = 1     # seconds
BUFFER_LENGTH = int(EPOCH_LENGTH * SAMPLING_RATE)
BANDS = {'Alpha': [8, 13], 'Beta': [13, 30]}

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data, axis=0)

def get_band_power(data, band, fs):
    freqs, psd = welch(data, fs, nperseg=fs)
    idx_band = np.logical_and(freqs >= BANDS[band][0], freqs <= BANDS[band][1])
    return np.sum(psd[idx_band])

def main():
    print("Looking for an EEG stream...")
    streams = resolve_stream('type', 'EEG')
    if not streams:
        print("🛑 No EEG stream found! Make sure EEGStudio is streaming via LSL.")
        return

    inlet = StreamInlet(streams[0])
    print("✅ Connected to EEG stream.")

    # CORRECTED LINE: Removed parentheses from inlet.channel_count
    eeg_buffer = np.zeros((BUFFER_LENGTH, inlet.channel_count))
    print("\n--- Starting Real-Time Analysis (Press Ctrl+C to stop) ---")

    try:
        while True:
            samples, _ = inlet.pull_chunk(timeout=1.5, max_samples=BUFFER_LENGTH)
            if samples:
                samples_np = np.array(samples)
                eeg_buffer = np.roll(eeg_buffer, -samples_np.shape[0], axis=0)
                eeg_buffer[-samples_np.shape[0]:, :] = samples_np
                
                channel_data = eeg_buffer[:, 0] # Analyzing the first channel
                filtered_data = bandpass_filter(channel_data, 1, 40, SAMPLING_RATE)
                
                alpha_power = get_band_power(filtered_data, 'Alpha', SAMPLING_RATE)
                beta_power = get_band_power(filtered_data, 'Beta', SAMPLING_RATE)

                print(f"Alpha Power: {alpha_power:.2f} | Beta Power: {beta_power:.2f}")
    
    except KeyboardInterrupt:
        print("\nStream reader stopped.")

if __name__ == '__main__':
    main()
