import os
import pandas as pd

# Base directory where all the P, W, A folders are located
base_dir = r'C:\Users\giaco\OneDrive\Desktop\Università\Tesi_Master\GitHub\Dataset'

def check_files_for_nans(base_dir):
    # Loop through P1 to P10
    for p in range(1, 11):
        # Loop through W1 to W4
        for w in range(1, 5):
            # Loop through A1
            for a in range(1, 2):
                # Construct the directory path
                dir_path = os.path.join(base_dir, f'P{p}', f'W{w}', f'A{a}', 'imu')
                # Construct the file path
                file_path = os.path.join(dir_path, 'merged_file_final.csv')
                
                # Check if the file exists
                if os.path.exists(file_path):
                    print(f'Checking file: {file_path}')
                    # Read the CSV file
                    df = pd.read_csv(file_path, header=None)
                    # Check for NaN values
                    if df.isnull().values.any():
                        print(f'Found NaN in file: {file_path}')

# Call the function
check_files_for_nans(base_dir)
