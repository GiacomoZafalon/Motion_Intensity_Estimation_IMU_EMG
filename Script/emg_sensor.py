import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
import matplotlib.pyplot as plt

tot_person = 2
tot_weights = 1
tot_attempts = 2

for person in range(1, tot_person + 1):
    for weight in range(1, tot_weights + 1):
        for attempt in range(1, tot_attempts + 1):
            # print(f'Processing data for Person {person}, Weight {weight}, Attempt {attempt}...')

            # File paths
            file_paths = [
                f'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/P{person}/W{weight}/A{attempt}/emg/emg_1.csv',
                f'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/P{person}/W{weight}/A{attempt}/emg/emg_2.csv',
                f'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/P{person}/W{weight}/A{attempt}/emg/emg_3.csv'
            ]

            mvc_path = f'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/P{person}/W{weight}/A{attempt}/emg/emg_mvc.csv'
            # Read the emg_mvc.csv file
            mvc_df = pd.read_csv(mvc_path)

            # Read and concatenate the data
            dfs = [pd.read_csv(file) for file in file_paths]
            merged_df = pd.concat(dfs, axis=1)

            # Save the merged DataFrame to a CSV file
            merged_df.to_csv('c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/emg_neural.csv', index=False)

            chan_1 = (merged_df.iloc[:, 0]).values
            chan_2 = (merged_df.iloc[:, 1]).values
            chan_3 = (merged_df.iloc[:, 2]).values
            mvc_sg = (mvc_df.iloc[:]).values.flatten()

            def apply_filters(signal, fs, lowcut_bp, highcut_bp, Q, f0, cutoff_freq):
                # Design the bandpass filter
                nyquist = 0.5 * fs
                low = lowcut_bp / nyquist
                high = highcut_bp / nyquist
                b_bp, a_bp = butter(order_bp, [low, high], btype='band')

                # Design the notch filter
                b_notch, a_notch = iirnotch(f0, Q, fs)

                # Apply the bandpass filter
                signal_filtered = filtfilt(b_bp, a_bp, signal)

                # Apply the notch filter
                signal_filtered = filtfilt(b_notch, a_notch, signal_filtered)

                # Rectify the signal
                signal_rectified = np.abs(signal_filtered)

                # Apply lowpass filter
                nyquist = 0.5 * fs
                normal_cutoff = cutoff_freq / nyquist
                b, a = butter(5, normal_cutoff, btype='low', analog=False)
                signal_lowpass_filtered = filtfilt(b, a, signal_rectified)

                # Compute RMS
                window_size = 5
                rms = np.sqrt(np.convolve(signal_lowpass_filtered**2, np.ones(window_size)/window_size, mode='valid'))
                pad_width = (window_size - 1) // 2
                rms = np.pad(rms, (pad_width, pad_width), mode='edge')

                return rms

            # Define filter parameters
            lowcut_bp = 5  # Hz
            highcut_bp = 45  # Hz
            order_bp = 6
            Q = 30.0  # Quality factor
            f0 = 50.0  # Center frequency
            cutoff_freq = 10  # Hz
            fs = 100 # Sampling frequency

            # Apply filters to chan_1
            chan_1_rms = apply_filters(chan_1, fs, lowcut_bp, highcut_bp, Q, f0, cutoff_freq)

            # Apply filters to chan_2
            chan_2_rms = apply_filters(chan_2, fs, lowcut_bp, highcut_bp, Q, f0, cutoff_freq)

            # Apply filters to chan_3
            chan_3_rms = apply_filters(chan_3, fs, lowcut_bp, highcut_bp, Q, f0, cutoff_freq)

            mvc_rms = apply_filters(mvc_sg, fs, lowcut_bp, highcut_bp, Q, f0, cutoff_freq)

            # Extract the maximum value
            mvc_max = mvc_rms.max()

            # Normalize the signals
            normalized_chan_1 = chan_1_rms / mvc_max
            normalized_chan_2 = chan_2_rms / mvc_max
            normalized_chan_3 = chan_3_rms / mvc_max

            # print(max(normalized_chan_1), max(normalized_chan_2), max(normalized_chan_3))

            label = max(normalized_chan_1) + max(normalized_chan_2) + max(normalized_chan_3)
            if label > 3:
                label = 3
            label = label / 3

            # print(label)

            # Create a DataFrame for the label
            label_df = pd.DataFrame({'label': [label]})

            # Save the DataFrame to a CSV file
            label_df.to_csv(f'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/P{person}/W{weight}/A{attempt}/emg/emg_label.csv')

    print(f'Data processing complete for Person {person}/{tot_person}')
