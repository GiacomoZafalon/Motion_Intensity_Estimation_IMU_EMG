import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import os
import random
import shutil

def add_gaussian_noise_to_quaternion(quaternion, std_dev):
    noise = np.random.normal(loc=0, scale=std_dev, size=4)
    noisy_quaternion = quaternion + noise
    return noisy_quaternion / np.linalg.norm(noisy_quaternion)

def time_warping(sensor_data, type=2, amount_of_warping=10):
    if type == 1:

        df = sensor_data

        # Define the interpolation function
        def interpolate_row(prev_row, next_row, num_interpolations=1):
            interpolated_rows = []
            for i in range(num_interpolations):
                alpha = (i + 1) / (num_interpolations + 1)
                interpolated_row = [prev_row[0]]  # Keep the first value (time) unchanged
                for j in range(1, len(prev_row)):
                    interpolated_value = prev_row[j] + (next_row[j] - prev_row[j]) * alpha
                    interpolated_row.append(interpolated_value)
                # Normalize the quaternion values
                quaternion_values = np.array(interpolated_row[-4:])
                norm = np.linalg.norm(quaternion_values)
                normalized_quaternion = quaternion_values / norm
                interpolated_row[-4:] = normalized_quaternion
                # interpolated_row.extend(normalized_quaternion)
                interpolated_rows.append(interpolated_row)
            return interpolated_rows

        # Initialize an empty DataFrame to store the interpolated data
        interpolated_data = []

        # Initialize a variable to track the time offset
        inter = 0

        # Iterate through the original DataFrame
        for i in range(len(df) - 1):
            append_value = df.iloc[i].values.tolist() + np.array([round(0.01*inter, 2)] + [0]*19)
            interpolated_data.append(append_value)
            if (i + 1) % amount_of_warping == 0:
                inter += 1
                interpolated_rows = interpolate_row(df.iloc[i].values.tolist(), df.iloc[i+1].values.tolist())
                for row in interpolated_rows:
                    interpolated_data.append(row + np.array([round(0.01*inter, 2)] + [0]*19))

        # Add the last row of the original DataFrame
        interpolated_data.append(df.iloc[-1].values.tolist() + np.array([round(0.01*inter, 2)] + [0]*19))

        # Create DataFrame from the interpolated data
        warped_df = pd.DataFrame(interpolated_data, columns=df.columns)
        # Round the timestamp column to 2 decimal places
        warped_df[0] = warped_df[0].round(2)

    elif type == 2:

        df = sensor_data

        contracted_time = []

        # Initialize a variable to track the time offset
        inter = 0

        # Iterate through the original DataFrame
        for i in range(len(df) - 1):
            if (i + 1) % amount_of_warping != 0:
                append_value = df.iloc[i].values.tolist() - np.array([round(0.01*inter, 2)] + [0]*19)
                contracted_time.append(append_value)
            else:
                inter += 1

        # Add the last row of the original DataFrame
        contracted_time.append(df.iloc[-1].values.tolist() - np.array([round(0.01*inter, 2)] + [0]*19))

        # Create DataFrame from the interpolated data
        warped_df = pd.DataFrame(contracted_time, columns=df.columns)
        # Round the timestamp column to 2 decimal places
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
                noisy_numbers = [numbers[0], str(float(numbers[1]) - noise)]
                # Write the noisy numbers back to the destination file
                destination_file_handle.write(','.join(noisy_numbers) + '\n')





tot_person = 1
tot_weights = 1
tot_attempts = 2
times = 1

for i in range(1, times + 1):
    for person in range(1, tot_person + 1):
        for weight in range(1, tot_weights + 1):
            for attempt in range(1, tot_attempts + 1):

                base_dir = f'C:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/'
                imu_data_path = os.path.join(base_dir, f'P{person}/W{weight}/A{attempt}/imu')
                emg_data_path = os.path.join(base_dir, f'P{person}/W{weight}/A{attempt}/emg')
                augmented_imu_data_dir = os.path.join(base_dir, f'P{person+i*tot_person}/W{weight}/A{attempt}/imu')
                augmented_emg_data_dir = os.path.join(base_dir, f'P{person+i*tot_person}/W{weight}/A{attempt}/emg')

                # List of sensor data files
                data_files = ['sensor1.csv', 'sensor2.csv', 'sensor3.csv', 'sensor4.csv']
                emg_file = 'emg_label.csv'

                warp_type = random.randint(1, 2) # 1-> Expansion, 2-> Contraction
                amount_of_warping = random.randint(10, 20) # Insert or delete a row every 10 to 20 timesteps (ms)

                mean_diff = []
                # Process each data file
                for file_name in data_files:
                    # Read the sensor data from the CSV file
                    imu_file_path = os.path.join(imu_data_path, file_name)
                    sensor_data = pd.read_csv(imu_file_path, header=None)

                    # Extract quaternions from columns 16 to 19
                    quaternions = sensor_data.iloc[:, 16:20].values

                    # Apply the Gaussian noise
                    augmented_quaternions = []
                    for quaternion in quaternions:
                        std_dev = 0.01
                        noisy_quaternion = add_gaussian_noise_to_quaternion(quaternion, std_dev)
                        augmented_quaternions.append(noisy_quaternion)

                    # Compute the difference between the original and the augmented quaternions
                    diff = augmented_quaternions - quaternions
                    mean_diff.append(np.mean(diff))
                    # Replace the original quaternions with augmented quaternions in the DataFrame
                    sensor_data.iloc[:, 16:20] = augmented_quaternions

                    # Apply the time warping
                    augmented_data = time_warping(sensor_data, warp_type, amount_of_warping)

                    # Ensure the output directory exists
                    os.makedirs(augmented_imu_data_dir, exist_ok=True)

                    # Write the augmented data back to the same CSV file
                    augmented_file_path = os.path.join(augmented_imu_data_dir, file_name.split('.')[0] + '.csv')
                    augmented_data.to_csv(augmented_file_path, index=False, header=False)

                # Find the mean value of noise throughout all 4 sensors
                mean_noise = np.mean(mean_diff) * 50

                add_noise_label(emg_data_path, augmented_emg_data_dir, mean_noise, 'emg_label.csv')

        print(f'Successfully augmented person {person}/{tot_person}')
    print(f'Round {i}/{times} completed')