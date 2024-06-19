import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
import matplotlib.pyplot as plt
import os

def apply_filters(signal, fs, lowcut_bp, highcut_bp, Q, f0, cutoff_freq, window):
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
    window_size = window
    rms = np.sqrt(np.convolve(signal_rectified**2, np.ones(window_size)/window_size, mode='valid'))
    pad_width = (window_size - 1) // 2
    rms = np.pad(rms, (pad_width, pad_width), mode='edge')

    return rms

def categorize_label(label):
    if 0 <= label <= 0.2:
        return 0
    elif 0.2 < label <= 0.4:
        return 1
    elif 0.4 < label <= 0.6:
        return 2
    elif 0.6 < label <= 0.8:
        return 3
    elif 0.8 < label <= 1.0:
        return 4
    else:
        print('Error in the label')

count_0 = 0
count_1 = 0
count_2 = 0
count_3 = 0
count_4 = 0

tot_person = 1
tot_weights = 5
tot_attempts = 1

for person in range(1, tot_person + 1):
    for weight in range(1, tot_weights + 1):
        for attempt in range(1, tot_attempts + 1):
            # File paths
            file_path = rf'C:\Users\giaco\OneDrive\Desktop\Università\Tesi_Master\GitHub\Dataset\P{person}\W{weight}\A{attempt}\emg\emg_data.csv'
            mvc_path = rf'C:\Users\giaco\OneDrive\Desktop\Università\Tesi_Master\GitHub\Dataset\P{person}\W5\A1\emg\emg_data.csv'

            if os.path.exists(file_path):

                # Read the emg_mvc.csv file
                mvc_df = pd.read_csv(mvc_path)

                # Read and concatenate the data
                dfs = pd.read_csv(file_path)

                chan_1 = (dfs.iloc[:, 0]).values
                chan_2 = (dfs.iloc[:, 1]).values
                chan_3 = (dfs.iloc[:, 2]).values
                mvc_1 = (mvc_df.iloc[:, 0]).values
                mvc_2 = (mvc_df.iloc[:, 1]).values
                mvc_3 = (mvc_df.iloc[:, 2]).values

                # Define filter parameters
                lowcut_bp = 0.1  # Hz
                highcut_bp = 45  # Hz
                order_bp = 6
                Q = 30.0  # Quality factor
                f0 = 50.0  # Center frequency
                cutoff_freq = 10  # Hz
                fs = 100 # Sampling frequency
                window_size = 5

                # Apply filters to chan_1
                chan_1_rms = apply_filters(chan_1, fs, lowcut_bp, highcut_bp, Q, f0, cutoff_freq, window_size)

                # Apply filters to chan_2
                chan_2_rms = apply_filters(chan_2, fs, lowcut_bp, highcut_bp, Q, f0, cutoff_freq, window_size)

                # Apply filters to chan_3
                chan_3_rms = apply_filters(chan_3, fs, lowcut_bp, highcut_bp, Q, f0, cutoff_freq, window_size)

                # Apply filter to mvc signal
                mvc_rms_1 = apply_filters(mvc_1, fs, lowcut_bp, highcut_bp, Q, f0, cutoff_freq, window_size)
                mvc_rms_2 = apply_filters(mvc_2, fs, lowcut_bp, highcut_bp, Q, f0, cutoff_freq, window_size)
                mvc_rms_3 = apply_filters(mvc_3, fs, lowcut_bp, highcut_bp, Q, f0, cutoff_freq, window_size)

                # Extract the maximum voluntary contruction (MVC)
                mvc_max = max(max(mvc_rms_1), max(mvc_rms_2), max(mvc_rms_3))

                # Normalize the signals
                normalized_chan_1 = chan_1_rms / mvc_max
                normalized_chan_2 = chan_2_rms / mvc_max
                normalized_chan_3 = chan_3_rms / mvc_max

                label = max(normalized_chan_1) + max(normalized_chan_2) + max(normalized_chan_3)
                if label > 3:
                    label = 3
                label = label / 3

            else:
                file_path = f'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/Dataset2/emg_label_p{person}_w{weight}_a{attempt}.csv'
                df_label = pd.read_csv(file_path)
                label = df_label.iloc[0, 0]
                if label > 1:
                    label = 1
                elif label < 0:
                    label = 0.05
                else: 
                    label = label

            # Categorize the label
            category = categorize_label(label)
            # Create a DataFrame for the label and category
            label_df = pd.DataFrame({'label': [label], 'class': [category]})
            print(label, category)
            if category == 0:
                count_0 += 1
            elif category == 1:
                count_1 += 1
            elif category == 2:
                count_2 += 1
            elif category == 3:
                count_3 += 1
            elif category == 4:
                count_4 += 1

            # Save the DataFrame to a CSV file
            label_df.to_csv(rf'C:\Users\giaco\OneDrive\Desktop\Università\Tesi_Master\GitHub\Dataset\P{person}\W{weight}\A{attempt}\emg\emg_label.csv', index=False)

    print(f'Data processing complete for Person {person}/{tot_person}')

print('count 0: ', count_0)
print('count 1: ', count_1)
print('count 2: ', count_2)
print('count 3: ', count_3)
print('count 4: ', count_4)
