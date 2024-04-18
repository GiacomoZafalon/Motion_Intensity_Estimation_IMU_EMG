import opensim as osim
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
import shutil
import numpy as np
from scipy.signal import butter, filtfilt

print('Processing data...')

attempt = 1
person = 1
weight = 1

# Directory where CSV files are located
data_dir = f'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/P{person}/W{weight}/A{attempt}/imu'

# List of CSV files
csv_files = ['sensor1.csv', 'sensor2.csv', 'sensor3.csv', 'sensor4.csv']

# Directory to store new CSV files
output_dir = f'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/P{person}/W{weight}/A{attempt}/imu'

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

# Define the paths of the input CSV files for each sensor
sensor_files = {
    'sensor1': f'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/P{person}/W{weight}/A{attempt}/imu/sensor1_sync_calib.csv',
    'sensor2': f'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/P{person}/W{weight}/A{attempt}/imu/sensor2_sync_calib.csv',
    'sensor3': f'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/P{person}/W{weight}/A{attempt}/imu/sensor3_sync_calib.csv',
    'sensor4': f'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/P{person}/W{weight}/A{attempt}/imu/sensor4_sync_calib.csv'
}

# Define the path of the output merged CSV file
merged_file = f'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/P{person}/W{weight}/A{attempt}/imu/merged_data.csv'

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

# File paths
quaternion_table_path = f'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/P{person}/W{weight}/A{attempt}/imu/quaternion_table.csv'

# Indices of columns to keep
columns_to_keep = [0, 10, 11, 12, 13, 24, 25, 26, 27, 38, 39, 40, 41, 52, 53, 54, 55]

# Read data from the input file and write filtered data to the output file
with open(merged_file, mode='r') as input_file, open(quaternion_table_path, mode='w', newline='') as output_file:
    reader = csv.reader(input_file)
    writer = csv.writer(output_file)
    
    for row in reader:
        filtered_row = [row[i] for i in columns_to_keep]
        writer.writerow(filtered_row)

# Input and output file paths
input_file_path = quaternion_table_path
output_file_path = f'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/P{person}/W{weight}/A{attempt}/imu/lifting_orientations.sto'
output_file_path_2 = 'c:/Users/giaco/Documents/OpenSim/4.5/Code/Python/OpenSenseExample/lifting_orientations.sto'

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

# Check if the file exists before attempting to delete it
if os.path.exists(output_file_path_2):
    # Delete the file
    os.remove(output_file_path_2)

# Write the modified data to the output file
with open(output_file_path_2, mode='w') as file:
    file.write(header)
    file.writelines(data_lines)

# File path to be deleted
file_to_delete = quaternion_table_path

# Check if the file exists before attempting to delete it
if os.path.exists(file_to_delete):
    # Delete the file
    os.remove(file_to_delete)

# List of files to delete
files_to_delete = [
    'sensor1_sync.csv',
    'sensor2_sync.csv',
    'sensor3_sync.csv',
    'sensor4_sync.csv',
    'sensor1_sync_calib.csv',
    'sensor2_sync_calib.csv',
    'sensor3_sync_calib.csv',
    'sensor4_sync_calib.csv'
]

# Iterate over the list of files and delete them
for file_name in files_to_delete:
    file_path = os.path.join(data_dir, file_name)
    if os.path.exists(file_path):
        os.remove(file_path)

print('Done... Ready for OpenSim')

# Setup and run the IMUPlacer tool, with model visualization set to true
imu_placer = osim.IMUPlacer('c:/Users/giaco/Documents/OpenSim/4.5/Code/Python/OpenSenseExample/myIMUPlacer_Setup.xml')
imu_placer.run(False)

# Write the calibrated model to file
calibrated_model = imu_placer.getCalibratedModel()

# Setup and run the IMU IK tool with visualization set to true
imu_ik = osim.IMUInverseKinematicsTool('c:/Users/giaco/Documents/OpenSim/4.5/Code/Python/OpenSenseExample/myIMUIK_Setup.xml')
imu_ik.run(False)

print('Computing and saving the motion data...')

# Read the motion data file into a DataFrame
motion_data = pd.read_csv('c:/Users/giaco/Documents/OpenSim/4.5/Code/Python/OpenSenseExample/IKResults/ik_lifting_orientations.mot', delimiter='\t', skiprows=6)

def calculate_angular_velocity(angles, timestamps):
    velocities = []
    velocities.append(0)
    for i in range(1, len(angles)):
        delta_angle = angles[i] - angles[i-1]
        delta_time = timestamps[i] - timestamps[i-1]
        velocity = delta_angle / delta_time
        velocities.append(velocity)
    return velocities

def calculate_angular_acceleration(velocities, timestamps):
    accelerations = []
    accelerations.append(0)
    for i in range(1, len(velocities)):
        delta_velocity = velocities[i] - velocities[i-1]
        delta_time = timestamps[i] - timestamps[i-1]
        acceleration = delta_velocity / delta_time
        accelerations.append(acceleration)
    return accelerations

time = motion_data['time']
shoulder_flex_angle = motion_data['arm_flex_r']
shoulder_add_angle = motion_data['arm_add_r']
elbow_flex_angle = motion_data['elbow_flex_r']
pelvis_angle = motion_data['pelvis_tilt']

def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_lowpass_filter(data, cutoff_frequency, sampling_frequency, filter_order=5):
    b, a = butter_lowpass(cutoff_frequency, sampling_frequency, order=filter_order)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

sampling_frequency = 100

cutoff_frequency_pos = 3
elbow_flex_angle_filt = apply_lowpass_filter(elbow_flex_angle, cutoff_frequency_pos, sampling_frequency)
shoulder_flex_angle_filt = apply_lowpass_filter(shoulder_flex_angle, cutoff_frequency_pos, sampling_frequency)
shoulder_add_angle_filt = apply_lowpass_filter(shoulder_add_angle, cutoff_frequency_pos, sampling_frequency)
pelvis_angle_filt = apply_lowpass_filter(pelvis_angle, cutoff_frequency_pos, sampling_frequency)

shoulder_flex_vel = calculate_angular_velocity(shoulder_flex_angle_filt, time)
shoulder_add_vel = calculate_angular_velocity(shoulder_add_angle_filt, time)
elbow_flex_vel = calculate_angular_velocity(elbow_flex_angle_filt, time)
pelvis_vel = calculate_angular_velocity(pelvis_angle_filt, time)

cutoff_frequency_vel = 3
elbow_flex_vel_filt = apply_lowpass_filter(elbow_flex_vel, cutoff_frequency_vel, sampling_frequency)
shoulder_flex_vel_filt = apply_lowpass_filter(shoulder_flex_vel, cutoff_frequency_vel, sampling_frequency)
shoulder_add_vel_filt = apply_lowpass_filter(shoulder_add_vel, cutoff_frequency_vel, sampling_frequency)
pelvis_vel_filt = apply_lowpass_filter(pelvis_vel, cutoff_frequency_vel, sampling_frequency)

shoulder_flex_acc = calculate_angular_acceleration(shoulder_flex_vel_filt, time)
shoulder_add_acc = calculate_angular_acceleration(shoulder_add_vel_filt, time)
elbow_flex_acc = calculate_angular_acceleration(elbow_flex_vel_filt, time)
pelvis_acc = calculate_angular_acceleration(pelvis_vel_filt, time)

cutoff_frequency_acc = 3
elbow_flex_acc_filt = apply_lowpass_filter(elbow_flex_acc, cutoff_frequency_acc, sampling_frequency)
shoulder_flex_acc_filt = apply_lowpass_filter(shoulder_flex_acc, cutoff_frequency_acc, sampling_frequency)
shoulder_add_acc_filt = apply_lowpass_filter(shoulder_add_acc, cutoff_frequency_acc, sampling_frequency)
pelvis_acc_filt = apply_lowpass_filter(pelvis_acc, cutoff_frequency_acc, sampling_frequency)

# Define the figure and subplots
fig, axs = plt.subplots(3, 4, figsize=(15, 15))

# Plot the filtered elbow angle
axs[0, 0].plot(time, elbow_flex_angle_filt, color='blue')
axs[0, 0].set_ylabel('Elbow Angle (degrees)')
axs[0, 0].set_title('Elbow Angle')
axs[0, 0].grid(True)

# Plot the filtered shoulder flexion angle
axs[0, 1].plot(time, shoulder_flex_angle_filt, color='green')
axs[0, 1].set_ylabel('Shoulder Flexion Angle (degrees)')
axs[0, 1].set_title('Shoulder Flexion Angle')
axs[0, 1].grid(True)

# Plot the filtered shoulder adduction angle
axs[0, 2].plot(time, shoulder_add_angle_filt, color='red')
axs[0, 2].set_ylabel('Shoulder Adduction Angle (degrees)')
axs[0, 2].set_title('Shoulder Adduction Angle')
axs[0, 2].grid(True)

# Plot the filtered shoulder adduction angle
axs[0, 3].plot(time, pelvis_angle_filt, color='black')
axs[0, 3].set_ylabel('Pelvis Angle (degrees)')
axs[0, 3].set_title('Pelvis Angle')
axs[0, 3].grid(True)

# Plot the filtered elbow velocity
axs[1, 0].plot(time, elbow_flex_vel_filt, color='blue')
axs[1, 0].set_ylabel('Elbow Velocity (degrees/s)')
axs[1, 0].set_title('Elbow Velocity')
axs[1, 0].grid(True)

# Plot the filtered shoulder flexion velocity
axs[1, 1].plot(time, shoulder_flex_vel_filt, color='green')
axs[1, 1].set_ylabel('Shoulder Flexion Velocity (degrees/s)')
axs[1, 1].set_title('Shoulder Flexion Velocity')
axs[1, 1].grid(True)

# Plot the filtered shoulder adduction velocity
axs[1, 2].plot(time, shoulder_add_vel_filt, color='red')
axs[1, 2].set_ylabel('Shoulder Adduction Velocity (degrees/s)')
axs[1, 2].set_title('Shoulder Adduction Velocity')
axs[1, 2].grid(True)

# Plot the filtered shoulder adduction angle
axs[1, 3].plot(time, pelvis_vel_filt, color='black')
axs[1, 3].set_ylabel('Pelvis Velocity (degrees/s)')
axs[1, 3].set_title('Pelvis Velocity')
axs[1, 3].grid(True)

# Plot the filtered elbow acceleration
axs[2, 0].plot(time, elbow_flex_acc_filt, color='blue')
axs[2, 0].set_ylabel('Elbow Acceleration (degrees/s^2)')
axs[2, 0].set_title('Elbow Acceleration')
axs[2, 0].grid(True)

# Plot the filtered shoulder flexion acceleration
axs[2, 1].plot(time, shoulder_flex_acc_filt, color='green')
axs[2, 1].set_ylabel('Shoulder Flexion Acceleration (degrees/s^2)')
axs[2, 1].set_title('Shoulder Flexion Acceleration')
axs[2, 1].grid(True)

# Plot the filtered shoulder adduction acceleration
axs[2, 2].plot(time, shoulder_add_acc_filt, color='red')
axs[2, 2].set_ylabel('Shoulder Adduction Acceleration (degrees/s^2)')
axs[2, 2].set_title('Shoulder Adduction Acceleration')
axs[2, 2].grid(True)

# Plot the filtered shoulder adduction angle
axs[2, 3].plot(time, pelvis_acc_filt, color='black')
axs[2, 3].set_ylabel('Pelvis Acceleration (degrees/s^2)')
axs[2, 3].set_title('Pelvis Acceleration')
axs[2, 3].grid(True)

# Set common xlabel
for ax in axs.flatten():
    ax.set_xlabel('Time')

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig(os.path.join(data_dir, 'joint_data.png'))

# # Show the plots
# plt.show()

# Define the file path for saving the CSV file
csv_file_path = os.path.join(data_dir, 'data_neural.csv')

# Define the data to be saved
data = {
    'time': time,
    'elbow_angle': elbow_flex_angle_filt,
    'shoulder_flex_angle': shoulder_flex_angle_filt,
    'shoulder_add_angle': shoulder_add_angle_filt,
    'pelvis_angle': pelvis_angle_filt,
    'elbow_velocity': elbow_flex_vel_filt,
    'shoulder_flex_velocity': shoulder_flex_vel_filt,
    'shoulder_add_velocity': shoulder_add_vel_filt,
    'pelvis_velocity': pelvis_vel_filt,
    'elbow_acceleration': elbow_flex_acc_filt,
    'shoulder_flex_acceleration': shoulder_flex_acc_filt,
    'shoulder_add_acceleration': shoulder_add_acc_filt,
    'pelvis_acceleration': pelvis_acc_filt
}

# Write the data to the CSV file
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header
    writer.writerow(data.keys())
    
    # Write the data rows
    for values in zip(*data.values()):
        writer.writerow(values)

print('Process complete')