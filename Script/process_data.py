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
        'linacc_z': [],
        'quat_w' : [],
        'quat_x' : [],
        'quat_y' : [],
        'quat_z' : []
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
    quat_w = [float(value) for value in columns['quat_w']]
    quat_x = [float(value) for value in columns['quat_x']]
    quat_y = [float(value) for value in columns['quat_y']]
    quat_z = [float(value) for value in columns['quat_z']]

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
    processed_file_path = f'{file_path[:-4]}_calib.csv'  # Append '_processed' to the original file name
    with open(processed_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['time', f'eul_z_{sensor_suffix}', f'eul_y_{sensor_suffix}', f'eul_x_{sensor_suffix}', 
                         f'gyro_x_{sensor_suffix}', f'gyro_y_{sensor_suffix}', f'gyro_z_{sensor_suffix}', 
                         f'linacc_x_{sensor_suffix}', f'linacc_y_{sensor_suffix}', f'linacc_z_{sensor_suffix}',
                         f'quat_w_{sensor_suffix}', f'quat_x_{sensor_suffix}', f'quat_y_{sensor_suffix}', f'quat_z_{sensor_suffix}'])
        for values in zip(time, eul_z, eul_y, eul_x, gyro_x, gyro_y, gyro_z, linacc_calib_x, linacc_calib_y, linacc_calib_z, quat_w, quat_x, quat_y, quat_z):
            writer.writerow(values)

    return processed_file_path

# Define calibration parameters for each sensor
calibration_params = {
    'sensor1': {'A': np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), 
                'b': np.array([0.0, 0.0, 0.0])},
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

print("Data has been syncronized and calibrated")

# Define the paths of the input CSV files for each sensor
sensor_files = {
    'sensor1': 'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/sensor1_sync_calib.csv',
    'sensor2': 'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/sensor2_sync_calib.csv',
    'sensor3': 'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/sensor3_sync_calib.csv',
    'sensor4': 'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/sensor4_sync_calib.csv'
}

# Define the path of the output merged CSV file
merged_file = 'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/merged_data.csv'

def merge_csv_files(sensor_files, merged_file):
    # Open the output CSV file for writing
    with open(merged_file, mode='w', newline='') as outfile:
        writer = csv.writer(outfile)

        # Open the input CSV file for sensor 1
        with open(sensor_files['sensor1'], mode='r') as file1:
            reader1 = csv.reader(file1)

            # Iterate through the rows of sensor 1
            for row1 in reader1:
                first_value = row1[0]

                # Look for matching rows in the other CSV files
                matching_row2 = find_matching_row(first_value, sensor_files['sensor2'])
                matching_row3 = find_matching_row(first_value, sensor_files['sensor3'])
                matching_row4 = find_matching_row(first_value, sensor_files['sensor4'])

                # If matching rows are found in all files, merge and write to the output CSV file
                if matching_row2 and matching_row3 and matching_row4:
                    merged_row = row1 + matching_row2 + matching_row3 + matching_row4
                    writer.writerow(merged_row)

def find_matching_row(value, filename):
    # Open the CSV file for reading
    with open(filename, mode='r') as file:
        reader = csv.reader(file)

        # Iterate through the rows of the CSV file
        for row in reader:
            # Check if the first value of the row matches the target value
            if row[0] == value:
                return row

    # If no matching row is found, return None
    return None

# Call the function to merge the CSV files
merge_csv_files(sensor_files, merged_file)

print('Merged data saved to: merged_data.csv')

# File paths
quaternion_table_path = 'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/quaternion_table.csv'

# Indices of columns to keep
columns_to_keep = [0, 10, 11, 12, 13, 24, 25, 26, 27, 38, 39, 40, 41, 52, 53, 54, 55]

# Read data from the input file and write filtered data to the output file
with open(merged_file, mode='r') as input_file, open(quaternion_table_path, mode='w', newline='') as output_file:
    reader = csv.reader(input_file)
    writer = csv.writer(output_file)
    
    for row in reader:
        filtered_row = [row[i] for i in columns_to_keep]
        writer.writerow(filtered_row)

print("Quaternion data has been written to: quaternion_table.csv")

# Input and output file paths
input_file_path = quaternion_table_path
output_file_path = 'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/lifting_orientations.sto'

# Read the data from the input file
with open(input_file_path, mode='r') as file:
    lines = file.readlines()

# Modify the header line
# Modify the header line
header = "DataRate=100.000000\n" \
         "DataType=Quaternion\n" \
         "version=3\n" \
         "OpenSimVersion=4.5-2024-01-10-34fd6af\n" \
         "endheader\n" \
         "time\tpelvis_imu\ttorso_imu\thumerus_r_imu\tulna_r_imu\n"

# Modify the data lines to replace comma with tab space for columns 0, 1, 5, and 6
data_lines = []
for line in lines[1:]:
    parts = line.split(",")
    modified_line = "\t".join(parts[:2]) + "," + ",".join(parts[2:5]) + "\t" + ",".join(parts[5:9]) + "\t" + ",".join(parts[9:13]) + "\t" + ",".join(parts[13:])
    data_lines.append(modified_line)

# Write the modified data to the output file
with open(output_file_path, mode='w') as file:
    file.write(header)
    file.writelines(data_lines)

print("Data has been written to: lifting_orientations.sto")

# File path to be deleted
file_to_delete = quaternion_table_path

# # Check if the file exists before attempting to delete it
# if os.path.exists(file_to_delete):
#     # Delete the file
#     os.remove(file_to_delete)
#     print("File quaternion_table.csv has been deleted.")
# else:
#     print("File quaternion_table.csv does not exist.")