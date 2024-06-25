import numpy as np
import pandas as pd
import os
import random
import shutil
from joblib import Parallel, delayed

def add_gaussian_noise_to_data(data, std_dev):
    noise = np.random.normal(loc=0, scale=std_dev, size=data.shape)
    noisy_data = np.round(data + noise, 2)
    return noisy_data

def time_warping(sensor_data, type=2, amount_of_warping=10):
    df = sensor_data.copy()
    if type == 1:
        def interpolate_row(prev_row, next_row, num_interpolations=1):
            interpolated_rows = []
            for i in range(num_interpolations):
                alpha = (i + 1) / (num_interpolations + 1)
                interpolated_row = prev_row + (next_row - prev_row) * alpha
                interpolated_rows.append(interpolated_row)
            return interpolated_rows

        interpolated_data = []
        inter = 0
        for i in range(len(df) - 1):
            interpolated_data.append(df.iloc[i].values)
            if (i + 1) % amount_of_warping == 0:
                inter += 1
                interpolated_rows = interpolate_row(df.iloc[i].values, df.iloc[i+1].values)
                interpolated_data.extend(interpolated_rows)
        interpolated_data.append(df.iloc[-1].values)
    elif type == 2:
        contracted_time = []
        inter = 0
        for i in range(len(df) - 1):
            if (i + 1) % amount_of_warping != 0:
                contracted_time.append(df.iloc[i].values)
            else:
                inter += 1
        contracted_time.append(df.iloc[-1].values)
        interpolated_data = contracted_time
    
    warped_df = pd.DataFrame(interpolated_data, columns=df.columns)
    return warped_df

def process_file(file_path, output_path, warp_type=2, amount_of_warping=10, std_dev=0.01):
    sensor_data = pd.read_csv(file_path, header=None)
    sensor_data = sensor_data.apply(pd.to_numeric)
    data = sensor_data.iloc[1:, 1:]
    augmented_data = time_warping(sensor_data, warp_type, amount_of_warping)

    augmented_values = augmented_data.iloc[:, 1:]
    noisy_values = add_gaussian_noise_to_data(augmented_values.values, std_dev)

    augmented_data.iloc[:, 1:] = noisy_values

    # Increment the first column by 0.01 starting from 0.01
    num_rows = len(augmented_data)
    augmented_data.iloc[:, 0] = np.round(np.arange(0.01, 0.01 * (num_rows + 1), 0.01)[:num_rows], 2)

    augmented_data.to_csv(output_path, index=False, header=None)

def augment_data(person, weight, attempt, time_step, base_dir, tot_person, tot_times, std_dev=0.1):
    imu_data_path = os.path.join(base_dir, f'P{person}/W{weight}/A{attempt}/imu/data_neural_euler_acc_gyro.csv')
    emg_data_path = os.path.join(base_dir, f'P{person}/W{weight}/A{attempt}/emg/emg_label.csv')
    
    augmented_base_dir = r'C:\Users\giaco\OneDrive\Desktop\Università\Tesi_Master\Dataset_train_augmented'
    augmented_imu_data_dir = os.path.join(augmented_base_dir, f'data_neural_euler_acc_gyro_P{person + time_step*tot_person}_W{weight}_A{attempt}.csv')
    augmented_emg_data_dir = os.path.join(augmented_base_dir, f'emg_label_P{person + time_step*tot_person}_W{weight}_A{attempt}.csv')

    warp_type = random.randint(1, 2)
    amount_of_warping = random.randint(10, 20)

    # Create necessary directories if they don't exist
    os.makedirs(os.path.dirname(augmented_imu_data_dir), exist_ok=True)
    os.makedirs(os.path.dirname(augmented_emg_data_dir), exist_ok=True)

    # Process the IMU data file
    process_file(imu_data_path, augmented_imu_data_dir, warp_type, amount_of_warping, std_dev)
    
    # Copy the EMG label file
    shutil.copy(emg_data_path, augmented_emg_data_dir)

    print(f'Completed augmentation for person {person}/{tot_person} for time {time_step}/{tot_times}')

# def process_person(person, tot_weights, tot_attempts, times, base_dir):
#     for time_step in range(1, times + 1):
#         for weight in range(1, tot_weights + 1):
#             for attempt in range(1, tot_attempts + 1):
#                 augment_data(person, weight, attempt, time_step, base_dir, tot_person, 0.1)
#         print(f'Completed augmentation for person {person}/{tot_person} for time {time_step}/{times}')

if __name__ == "__main__":
    tot_person = 11
    tot_weights = 5
    tot_attempts = 10
    times = 300

    base_dir = 'C:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset_train/'

    # Use joblib for parallel processing
    Parallel(n_jobs=-1)(
    delayed(augment_data)(person, weight, attempt, i, base_dir, tot_person, times, 0.1)
    for i in range(1, times + 1)
    for person in range(1, tot_person + 1)
    for weight in range(1, tot_weights + 1)
    for attempt in range(1, tot_attempts + 1)
)

print("Data augmentation complete.")
