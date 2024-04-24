import numpy as np
import pandas as pd
import os

person = 1
weight = 1
attempt = 1

original_data_dir = f'C:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/P{person}/W{weight}/A{attempt}/imu'
file_name = ['sensor1.csv', 'sensor2.csv', 'sensor3.csv', 'sensor4.csv']

for file in file_name:
    # Read the sensor data from the CSV file
    file_path = os.path.join(original_data_dir, file)
    data = pd.read_csv(file_path, header=None)

    # Interpolate missing timesteps
    new_data = []
    for i in range(len(data) - 1):
        new_data.append(data.iloc[i])
        current_time = round(data.iloc[i, 0], 2)
        next_time = round(data.iloc[i + 1, 0], 2)
        gap = round(next_time - current_time, 2)
        if gap > 0.01:  # Check if there is a jump
            num_missing_steps = int(gap / 0.01) - 1
            step_size = (data.iloc[i + 1] - data.iloc[i]) / (num_missing_steps + 1)
            new_data_s = []
            for j in range(1, num_missing_steps + 1):
                interpolated_step = data.iloc[i] + step_size * j
                new_data_s.append(interpolated_step)
            # Normalize the quaternion values
            quaternion_values = np.array(new_data_s[-4:])
            norm = np.linalg.norm(quaternion_values)
            normalized_quaternion = quaternion_values / norm
            new_data_s[-4:] = normalized_quaternion
            # interpolated_row.extend(normalized_quaternion)
            new_data.append(new_data_s)

    new_data.append(data.iloc[-1])  # Add the last row

    # Convert new data to DataFrame
    interpolated_df = pd.DataFrame(new_data)

    interpolated_df[0] = interpolated_df[0].round(2)

    # Save interpolated data to CSV file
    interpolated_file_path = os.path.join(original_data_dir, f'interpolated_{file}')
    interpolated_df.to_csv(interpolated_file_path, index=False, header=False)
