import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import os
import random

def add_gaussian_noise_to_quaternion(quaternion, std_dev):
    noise = np.random.normal(loc=0, scale=std_dev, size=4)
    noisy_quaternion = quaternion + noise
    return noisy_quaternion / np.linalg.norm(noisy_quaternion)

def time_warping(file_path, type=2, amount_of_warping=10):
    if type == 1:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path, header=None)

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
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path, header=None)

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

tot_person = 1
tot_weights = 1
tot_attempts = 2
num_real_people = 1

for person in range(1, tot_person + 1):
    for weight in range(1, tot_weights + 1):
        for attempt in range(1, tot_attempts + 1):

            original_data_dir = f'C:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/P{person}/W{weight}/A{attempt}/imu'
            augmented_data_dir = f'C:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/P{person+num_real_people}/W{weight}/A{attempt}/imu'

            # List of sensor data files
            data_files = ['sensor1.csv', 'sensor2.csv', 'sensor3.csv', 'sensor4.csv']

            # rand = random.randint(1, 4)
            rand = 2
            if rand == 1:
                print('Data augmentation through noise')
            if rand == 2:
                # warp_type = random.randint(1, 2)
                warp_type = 2
                amount_of_warping = random.randint(10, 20)
                if warp_type == 1:
                    print('Data augmentation through time warping - Expansion')
                else:
                    print('Data augmentation through time warping - Contraction')

            # Process each data file
            for file_name in data_files:
                # Read the sensor data from the CSV file
                file_path = os.path.join(original_data_dir, file_name)
                sensor_data = pd.read_csv(file_path, header=None)

                # Extract quaternions from columns 10 to 13
                quaternions = sensor_data.iloc[:, 16:20].values

                # Perform data augmentation on quaternions
                augmented_quaternions = []
                for quaternion in quaternions:
                    # Generate noisy quaternion
                    if rand == 1:
                        std_dev = 0.01
                        noisy_quaternion = add_gaussian_noise_to_quaternion(quaternion, std_dev)
                        # Replace the original quaternions with augmented quaternions in the DataFrame
                        sensor_data.iloc[:, 16:20] = augmented_quaternions
                    elif rand == 2:
                        sensor_data = time_warping(file_path, warp_type, amount_of_warping)
                    # elif rand == 3:
                    # elif rand == 4:
                    # augmented_quaternions.append(noisy_quaternion)

                

                # Ensure the output directory exists
                os.makedirs(augmented_data_dir, exist_ok=True)

                # Write the augmented data back to the same CSV file
                augmented_file_path = os.path.join(augmented_data_dir, file_name.split('.')[0] + '.csv')
                sensor_data.to_csv(augmented_file_path, index=False, header=False)

            print(f'Successfully augmented person {person}/{tot_person}, weight {weight}/{tot_weights}, attempt {attempt}/{tot_attempts}')