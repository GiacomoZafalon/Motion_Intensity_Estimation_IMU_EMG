import numpy as np
import pandas as pd
import os
import random
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

def add_noise_label(source_dir, destination_dir, noise):
    # if not os.path.exists(destination_dir):
    #     os.makedirs(destination_dir)
    
    source_file = source_dir
    destination_file = destination_dir

    with open(source_file, 'r') as source_file_handle:
        lines = source_file_handle.readlines()
        with open(destination_file, 'w') as destination_file_handle:
            destination_file_handle.write(lines[0])
            for line in lines[1:]:
                numbers = line.strip().split(',')
                noisy_numbers = [str(float(numbers[0]) - noise), numbers[1]]
                destination_file_handle.write(','.join(noisy_numbers) + '\n')

def process_file(file_path, output_dir, warp_type=2, amount_of_warping=10, std_dev=0.01):
    sensor_data = pd.read_csv(file_path, header=None)
    sensor_data = sensor_data.apply(pd.to_numeric)
    data = sensor_data.iloc[1:, 1:]
    augmented_data = time_warping(sensor_data, warp_type, amount_of_warping)

    augmented_values = augmented_data.iloc[:, 1:]
    noisy_values = add_gaussian_noise_to_data(augmented_values.values, std_dev)

    diff = augmented_values.values - noisy_values
    mean_diff = np.mean(diff)

    mean_noise = np.mean(mean_diff) * 50
    augmented_data.iloc[:, 1:] = noisy_values
    augmented_data.to_csv(output_dir, index=False, header=False)

    return mean_noise

def augment_data(person, weight, attempt, time_step, tot_times, base_dir, tot_person, std_dev=0.1):
    imu_data_path = os.path.join(base_dir, f'data_neural_euler_acc_gyro_p{person}_w{weight}_a{attempt}.csv')
    emg_data_path = os.path.join(base_dir, f'emg_label_p{person}_w{weight}_a{attempt}.csv')
    augmented_imu_data_dir = os.path.join(base_dir, f'data_neural_euler_acc_gyro_p{person + time_step * tot_person}_w{weight}_a{attempt}.csv')
    augmented_emg_data_dir = os.path.join(base_dir, f'emg_label_p{person + time_step * tot_person}_w{weight}_a{attempt}.csv')

    # if not os.path.exists(augmented_imu_data_dir):
    #     os.makedirs(augmented_imu_data_dir)

    warp_type = random.randint(1, 2)
    amount_of_warping = random.randint(10, 20)

    file_paths = ['data_neural_euler_acc_gyro.csv']

    mean_noise = []
    # for file_path in file_paths:
    input_path = imu_data_path
    output_path = augmented_imu_data_dir
    noise_mean = process_file(input_path, output_path, warp_type, amount_of_warping, std_dev)
    mean_noise.append(noise_mean)

    final_noise = np.mean(mean_noise)
    add_noise_label(emg_data_path, augmented_emg_data_dir, final_noise)

    print(f'Completed augmentation for person {person}/{tot_person} for time {time_step}/{tot_times}')

if __name__ == "__main__":
    tot_person = 12
    tot_weights = 5
    tot_attempts = 6
    times = 900

    base_dir = 'C:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/Dataset2/'

    # Use joblib for parallel processing
    Parallel(n_jobs=-1)(
        delayed(augment_data)(person, weight, attempt, i, times, base_dir, tot_person)
        for i in range(2, times + 1)
        for person in range(1, tot_person + 1)
        for weight in range(1, tot_weights + 1)
        for attempt in range(1, tot_attempts + 1)
    )

    print("Data augmentation complete.")
