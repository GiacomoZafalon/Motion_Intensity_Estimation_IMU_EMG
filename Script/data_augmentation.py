import numpy as np
import pandas as pd
import os
import random

def add_gaussian_noise_to_data(data, std_dev):
    noise = np.random.normal(loc=0, scale=std_dev, size=len(data))
    # print(data)
    # print(noise)
    noisy_data = data + noise
    return noisy_data

def time_warping(sensor_data, type=2, amount_of_warping=10):
    if type == 1:
        df = sensor_data
        def interpolate_row(prev_row, next_row, num_interpolations=1):
            interpolated_rows = []
            for i in range(num_interpolations):
                alpha = (i + 1) / (num_interpolations + 1)
                interpolated_row = [prev_row[0]]
                for j in range(1, len(prev_row)):
                    interpolated_value = prev_row[j] + (next_row[j] - prev_row[j]) * alpha
                    interpolated_row.append(interpolated_value)
                # quaternion_values = np.array(interpolated_row[-4:])
                # norm = np.linalg.norm(quaternion_values)
                # normalized_quaternion = quaternion_values / norm
                # interpolated_row[-4:] = normalized_quaternion
                interpolated_rows.append(interpolated_row)
            return interpolated_rows
        interpolated_data = []
        inter = 0
        for i in range(len(df) - 1):
            append_value = df.iloc[i].values.tolist() + np.array([round(0.01*inter, 2)] + [0]*(df.shape[1]-1))
            interpolated_data.append(append_value)
            if (i + 1) % amount_of_warping == 0:
                inter += 1
                interpolated_rows = interpolate_row(df.iloc[i].values.tolist(), df.iloc[i+1].values.tolist())
                for row in interpolated_rows:
                    interpolated_data.append(row + np.array([round(0.01*inter, 2)] + [0]*(df.shape[1]-1)))
        interpolated_data.append(df.iloc[-1].values.tolist() + np.array([round(0.01*inter, 2)] + [0]*(df.shape[1]-1)))
        warped_df = pd.DataFrame(interpolated_data, columns=df.columns)
        warped_df[0] = warped_df[0].round(2)
    elif type == 2:
        df = sensor_data
        # print(df.shape[1])
        contracted_time = []
        inter = 0
        for i in range(len(df) - 1):
            if (i + 1) % amount_of_warping != 0:
                append_value = df.iloc[i].values.tolist() - np.array([round(0.01*inter, 2)] + [0]*(df.shape[1]-1))
                contracted_time.append(append_value)
            else:
                inter += 1
        contracted_time.append(df.iloc[-1].values.tolist() - np.array([round(0.01*inter, 2)] + [0]*(df.shape[1]-1)))
        warped_df = pd.DataFrame(contracted_time, columns=df.columns)
        warped_df[0] = warped_df[0].round(2)
    return warped_df

def add_noise_label(source_dir, destination_dir, noise, filename):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    
    # Construct the full paths for the source and destination files
    source_file = os.path.join(source_dir, filename)
    destination_file = os.path.join(destination_dir, filename)

    # Read the source file and write modified contents to the destination file
    with open(source_file, 'r') as source_file_handle:
        lines = source_file_handle.readlines()
        # Write the first line (header) to the destination file
        with open(destination_file, 'w') as destination_file_handle:
            destination_file_handle.write(lines[0])
            # Process and write the remaining lines (data)
            for line in lines[1:]:
                # Split the line into numbers
                numbers = line.strip().split(',')
                # Convert the numbers to floats and add noise
                # If noise > 0 it means that the angles are slightly bigger, so the load was not that heavy
                noisy_numbers = [str(float(numbers[0]) - noise), numbers[1]]
                # Write the noisy numbers back to the destination file
                destination_file_handle.write(','.join(noisy_numbers) + '\n')

def process_file(file_path, output_dir, warp_type=2, amount_of_warping=10, std_dev=0.01):
    sensor_data = pd.read_csv(file_path, header=None)
    sensor_data = sensor_data.apply(pd.to_numeric)
    data = sensor_data.iloc[1:, 1:].values
    augmented_data = time_warping(sensor_data, warp_type, amount_of_warping)

    # Extract the values from augmented_data excluding the first column (timestamps)
    augmented_values = augmented_data.iloc[:, 1:].values

    # Generate noise for each value in augmented_data excluding the timestamps
    noisy_values = np.array([add_gaussian_noise_to_data(d, std_dev) for d in augmented_values])

    # Assign the noisy values back to the augmented_data DataFrame
    augmented_data.iloc[:, 1:] = noisy_values

    # Calculate the difference between augmented_data and original data
    diff = augmented_values - noisy_values
    mean_diff = np.mean(diff)

    mean_noise = np.mean(mean_diff) * 50
    augmented_data.to_csv(output_dir, index=False, header=False)

    return mean_noise







tot_person = 5
tot_weights = 5
tot_attempts = 1
times = 5


for person in range(1, tot_person + 1):
    for weight in range(1, tot_weights + 1):
        for attempt in range(1, tot_attempts + 1):
            for i in range(1, times + 1):

                base_dir = f'C:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/'
                imu_data_path = os.path.join(base_dir, f'P{person}/W{weight}/A{attempt}/imu')
                emg_data_path = os.path.join(base_dir, f'P{person}/W{weight}/A{attempt}/emg')
                # augmented_imu_data_dir = os.path.join(base_dir, f'P{person+i*tot_person}/W{weight}/A{attempt}/imu')
                # augmented_emg_data_dir = os.path.join(base_dir, f'P{person+i*tot_person}/W{weight}/A{attempt}/emg')
                augmented_imu_data_dir = os.path.join(base_dir, f'P{person}/W{weight}/A{attempt+i}/imu')
                augmented_emg_data_dir = os.path.join(base_dir, f'P{person}/W{weight}/A{attempt+i}/emg')

                if not os.path.exists(augmented_imu_data_dir):
                    os.makedirs(augmented_imu_data_dir)
                
                warp_type = random.randint(1, 2) # 1-> Expansion, 2-> Contraction
                amount_of_warping = random.randint(10, 20) # Insert or delete a row every 10 to 20 timesteps (ms)

                std_dev = 0.1

                file_paths = [
                    'data_neural.csv',
                    'merged_data.csv'
                ]

                mean_noise = []
                for file_path in file_paths:
                    input_path = os.path.join(imu_data_path, file_path)
                    output_path = os.path.join(augmented_imu_data_dir, file_path)
                    noise_mean = process_file(input_path, output_path, warp_type, amount_of_warping, std_dev)
                    mean_noise.append(noise_mean)
                    
                final_noise = np.mean(mean_noise)
                add_noise_label(emg_data_path, augmented_emg_data_dir, final_noise, 'emg_label.csv')

                print(f'Augmented person {person}/{tot_person}, weight {weight}/{tot_weights} for time {i}/{times}')

print("Data augmentation complete.")
