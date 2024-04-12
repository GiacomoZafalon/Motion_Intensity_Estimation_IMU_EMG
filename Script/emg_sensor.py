import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
import matplotlib.pyplot as plt


file_path_raw = 'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/Jupyter/40_raw.csv'
file_path_fil = 'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/Jupyter/40_filtered.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path_raw)
df_fil = pd.read_csv(file_path_fil)

print(df.head())

chan_1 = (df.iloc[:, 0]).values
chan_2 = (df.iloc[:, 1]).values
chan_3 = (df.iloc[:, 2]).values
chan_4 = (df.iloc[:, 3]).values

chan_1_fil = (df_fil.iloc[:, 0]).values
chan_2_fil = (df_fil.iloc[:, 1]).values
chan_3_fil = (df_fil.iloc[:, 2]).values
chan_4_fil = (df_fil.iloc[:, 3]).values

fig, axs = plt.subplots(2, 2)

axs[0, 0].plot(chan_1)
axs[0, 0].set_title('chan 1')

axs[0, 1].plot(chan_2)
axs[0, 1].set_title('chan 2')

axs[1, 0].plot(chan_3)
axs[1, 0].set_title('chan 3')

axs[1, 1].plot(chan_4)
axs[1, 1].set_title('chan 4')

fig, axs = plt.subplots(2, 2)

axs[0, 0].plot(chan_1_fil)
axs[0, 0].set_title('chan 1')

axs[0, 1].plot(chan_2_fil)
axs[0, 1].set_title('chan 2')

axs[1, 0].plot(chan_3_fil)
axs[1, 0].set_title('chan 3')

axs[1, 1].plot(chan_4_fil)
axs[1, 1].set_title('chan 4')

def butter_bandpass(lowcut, highcut, fs, order=6):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def notch_filter(fs, Q, f0):
    b, a = iirnotch(f0, Q, fs)
    return b, a

def apply_filters(signal, fs):
    # Define the bandpass filter parameters
    lowcut_bp = 5
    highcut_bp = 500
    order_bp = 6

    # Define the notch filter parameters
    Q = 30.0  # Quality factor
    f0 = 50.0  # Center frequency

    # Design the bandpass filter
    b_bp, a_bp = butter_bandpass(lowcut_bp, highcut_bp, fs, order=order_bp)

    # Design the notch filter
    b_notch, a_notch = notch_filter(fs, Q, f0)

    # Apply the bandpass filter
    signal_filtered = filtfilt(b_bp, a_bp, signal)

    # Apply the notch filter
    signal_filtered = filtfilt(b_notch, a_notch, signal_filtered)

    return signal_filtered

# Example usage:
# Assuming chan_1 is your signal and fs is the sampling frequency
fs = 2000
chan_1_filtered = apply_filters(chan_1, fs)
plt.plot(chan_1_filtered)

chan_1_rec = abs(chan_1_filtered)
plt.plot(chan_1_rec)

def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# Define the cutoff frequency for the lowpass filter
cutoff_freq = 2  # Adjust as needed

# Define the sampling frequency (assuming fs is your sampling frequency)
fs = 2000  # Adjust as needed

# Apply the lowpass filter to chan_1_filtered
chan_1_lowpass_filtered = lowpass_filter(chan_1_rec, cutoff_freq, fs)
plt.plot(chan_1_lowpass_filtered)

def smooth_with_rms(signal, window_size):
    # Compute the RMS over a sliding window
    rms = np.sqrt(np.convolve(signal**2, np.ones(window_size)/window_size, mode='valid'))
    
    # Pad the RMS array to have the same length as the original signal
    pad_width = (window_size - 1) // 2
    rms = np.pad(rms, (pad_width, pad_width), mode='edge')
    
    return rms

# Define the window size for computing RMS (choose an appropriate value)
window_size = 500

# Smooth the EMG signal using RMS
chan_1_rms = smooth_with_rms(chan_1_lowpass_filtered, window_size)
plt.plot(chan_1_rms)