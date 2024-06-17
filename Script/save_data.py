import pandas as pd
import numpy as np
import os
import shutil

# Total counts for persons, weights, and attempts
tot_person = 1
tot_weight = 1
tot_attempt = 1

# Source and destination folders
source_folder = 'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/Dataset2'

# Function to process CSV files: interpolate missing values
def interpolate_csv(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Replace 0s with NaNs
    df.replace(0, np.nan, inplace=True)

    # Interpolate missing values
    df.interpolate(method='linear', inplace=True)

    # Save the interpolated data back to the same file
    df.to_csv(file_path, index=False)

# Iterate through all person, weight, and attempt combinations
for person in range(1, tot_person + 1):
    for weight in range(1, tot_weight + 1):
        for attempt in range(1, tot_attempt + 1):

            person = 10812
            weight = 5
            attempt = 6
            # Define paths to IMU and EMG folders
            imu_path = f'{source_folder}/data_neural_euler_acc_gyro_p{person}_w{weight}_a{attempt}.csv'

            # Define the IMU CSV file path
            data_neural_file = imu_path

            # Check if the IMU CSV file exists
            if os.path.exists(data_neural_file):
                # Interpolate missing values in the IMU CSV file
                interpolate_csv(data_neural_file)



    print(f'person {person}/{tot_person} done')

print('Done')
