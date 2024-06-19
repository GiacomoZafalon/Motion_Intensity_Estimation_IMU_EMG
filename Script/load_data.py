import os
import pandas as pd
import re

# Directory where the files are located
root_dir = r'C:\Users\giaco\OneDrive\Desktop\Università\Tesi_Master\Dataset2'  # Replace with the root directory of your dataset

# Regex pattern to match filenames
pattern = re.compile(r'data_neural_euler_acc_gyro_p\d+_w\d+_a\d+\.csv')

def check_files_for_nans(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if pattern.match(file):
                file_path = os.path.join(root, file)
                print(file_path)
                df = pd.read_csv(file_path)
                if df.isnull().values.any():
                    print(f'Found NaN in file: {file_path}')
                    print(1)

# Call the function
check_files_for_nans(root_dir)
