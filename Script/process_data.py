import os
import csv
import shutil
import numpy as np

# Directory where CSV files are located
data_dir = 'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset'

# List of CSV files
csv_files = ['sensor1.csv', 'sensor2.csv', 'sensor3.csv', 'sensor4.csv']

# Directory to store new CSV files
output_dir = 'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# List to store last values from the first column of each CSV file
last_values = []

# Read the last value from the first column of each CSV file and create new files
for csv_file in csv_files:
    input_file_path = os.path.join(data_dir, csv_file)
    output_file_path = os.path.join(output_dir, csv_file.replace('.csv', '_sync.csv'))
    
    with open(input_file_path, mode='r') as input_file, open(output_file_path, mode='w', newline='') as output_file:
        reader = csv.reader(input_file)
        writer = csv.writer(output_file)
        
        # Copy the rows from input to output, updating the first column
        for row in reader:
            first_value = float(row[0])
            new_first_value = first_value
            row[0] = str(round(new_first_value, 2))
            writer.writerow(row)
    
    # Append the last value to the list
    last_values.append(new_first_value)

# Convert last values to integers
last_values = [int(x * 100) for x in last_values]

# Find the lowest value among the last values
min_value = min(last_values)

# Calculate the differences between each last value and the lowest value
differences = [value - min_value for value in last_values]

# Indices of columns to be removed (0-indexed)
columns_to_remove = [4, 5, 6, 10, 11, 12]

# Remove specified columns from the newly created CSV files
for csv_file in csv_files:
    output_file_path = os.path.join(output_dir, csv_file.replace('.csv', '_sync.csv'))
    
    with open(output_file_path, mode='r') as file:
        lines = file.readlines()
        
    with open(output_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write rows back to the file, removing specified columns
        for line in lines:
            values = line.strip().split(',')
            updated_values = [value for index, value in enumerate(values) if index not in columns_to_remove]
            writer.writerow(updated_values)

def process_file(file_path, sensor_suffix, A, b):
    columns = {  # Dictionary to store each column's data
        'time': [],
        'eul_z': [],
        'eul_y': [],
        'eul_x': [],
        'gyro_x': [],
        'gyro_y': [],
        'gyro_z': [],
        'linacc_x': [],
        'linacc_y': [],
        'linacc_z': []
    }

    with open(file_path, mode='r', newline='') as file:
        reader = csv.DictReader(file, fieldnames=columns.keys())
        next(reader)  # Skip the header row
        for row in reader:
            for key, value in row.items():
                columns[key].append(value)

    # Assign each column to a variable
    time = [float(value) for value in columns['time']]
    eul_z = [float(value) for value in columns['eul_z']]
    eul_y = [float(value) for value in columns['eul_y']]
    eul_x = [float(value) for value in columns['eul_x']]
    gyro_x = [float(value) for value in columns['gyro_x']]
    gyro_y = [float(value) for value in columns['gyro_y']]
    gyro_z = [float(value) for value in columns['gyro_z']]
    linacc_x = [float(value) for value in columns['linacc_x']]
    linacc_y = [float(value) for value in columns['linacc_y']]
    linacc_z = [float(value) for value in columns['linacc_z']]

    # Define calibration parameters
    # Convert the lists to NumPy arrays
    linacc_x = np.array(linacc_x)
    linacc_y = np.array(linacc_y)
    linacc_z = np.array(linacc_z)

    # Stack the arrays column-wise to create the matrix
    rawAcc = np.column_stack((linacc_x, linacc_y, linacc_z))
    N = len(rawAcc)
    calibAcc = np.zeros((N, 3), dtype='float')
    for i in range(N):
        currMeas = np.array([rawAcc[i, 0], rawAcc[i, 1], rawAcc[i, 2]])
        calibAcc[i, :] = A @ (currMeas - b)

    # Create lists for calibrated acceleration components
    linacc_calib_x = calibAcc[:, 0].tolist()
    linacc_calib_y = calibAcc[:, 1].tolist()
    linacc_calib_z = calibAcc[:, 2].tolist()

    # Write the processed data back to the CSV file
    processed_file_path = f'{file_path[:-9]}_calib.csv'  # Append '_processed' to the original file name
    with open(processed_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['time', f'eul_z_{sensor_suffix}', f'eul_y_{sensor_suffix}', f'eul_x_{sensor_suffix}', 
                         f'gyro_x_{sensor_suffix}', f'gyro_y_{sensor_suffix}', f'gyro_z_{sensor_suffix}', 
                         f'linacc_x_{sensor_suffix}', f'linacc_y_{sensor_suffix}', f'linacc_z_{sensor_suffix}'])
        for values in zip(time, eul_z, eul_y, eul_x, gyro_x, gyro_y, gyro_z, linacc_calib_x, linacc_calib_y, linacc_calib_z):
            writer.writerow(values)

    return processed_file_path

# Define calibration parameters for each sensor
calibration_params = {
    'sensor1': {'A': np.array([[1.004332, 0.000046, 0.004896], [0.000046, 0.969793, 0.009452], [0.004896, 0.009452, 1.022384]]), 
                'b': np.array([0.027031, -0.040204, 0.046558])},
    'sensor2': {'A': np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), 
                'b': np.array([0.0, 0.0, 0.0])},
    'sensor3': {'A': np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), 
                'b': np.array([0.0, 0.0, 0.0])},
    'sensor4': {'A': np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), 
                'b': np.array([0.0, 0.0, 0.0])}
}

# Process each sensor data
for sensor_suffix, params in calibration_params.items():
    file_path = os.path.join(data_dir, f'{sensor_suffix}_sync.csv')
    process_file(file_path, sensor_suffix, params['A'], params['b'])

print('Processing complete.')