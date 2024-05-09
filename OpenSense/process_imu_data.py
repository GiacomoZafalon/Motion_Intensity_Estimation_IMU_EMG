import opensim as osim
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
import numpy as np
from scipy.signal import butter, filtfilt

# Define the base directory where all data is located
base_dir = 'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/'

tot_person = 1
tot_weights = 1
tot_attempts = 1

# person = 2
# weight = 1
# attempt = 1

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - z1 * z2 - y1 * y2
    x = w1 * x2 + x1 * w2 - z1 * y2 + y1 * z2
    y = w1 * y2 + x1 * z2 + z1 * w2 - y1 * x2
    z = w1 * z2 - x1 * y2 + z1 * w2 + y1 * x2
    return np.array([w, x, y, z])

def rotate_up_low_arm(data_dir, angle_x, angle_y, angle_z, file_names):
    for file in file_names[2:]:
        data_file_path = os.path.join(data_dir, file)
        data = pd.read_csv(data_file_path)
        quaternion_columns = data.iloc[:, -4:]

        # Define the rotation quaternions for 180-degree rotation around the x-axis and z-axis
        rotation_quaternion_x = np.array([np.cos(angle_x / 2), np.sin(angle_x / 2), 0, 0])
        rotation_quaternion_y = np.array([np.cos(angle_y / 2), 0, np.sin(angle_y / 2), 0])
        rotation_quaternion_z = np.array([np.cos(angle_z / 2), 0, 0, np.sin(angle_z / 2)])

        # Rotate each quaternion in the set
        rotated_quaternions = []
        for i in range(len(quaternion_columns)):
            quaternion = quaternion_columns.iloc[i].to_numpy()
            rotated_quaternion_x = quaternion_multiply(rotation_quaternion_x, quaternion)
            rotated_quaternion_xy = quaternion_multiply(rotation_quaternion_y, rotated_quaternion_x)
            rotated_quaternion_xyz = quaternion_multiply(rotation_quaternion_z, rotated_quaternion_xy)
            rotated_quaternions.append(rotated_quaternion_xyz)

        # Update the corresponding columns in the DataFrame with the rotated quaternions
        for i, col in enumerate(quaternion_columns.columns):
            data[col] = [q[i] for q in rotated_quaternions]

        # Save the updated DataFrame to the output CSV file
        data.to_csv(data_file_path.replace('.csv', '_rotated.csv'), index=False, header=False)

def rotate_pelvis_torso(data_dir, angle_x, angle_y, angle_z, file_names):
    for file in file_names[:2]:
        data_file_path = os.path.join(data_dir, file)
        data = pd.read_csv(data_file_path)
        quaternion_columns = data.iloc[:, -4:]

        # Define the rotation quaternions for 180-degree rotation around the x-axis and z-axis
        rotation_quaternion_x = np.array([np.cos(angle_x / 2), np.sin(angle_x / 2), 0, 0])
        rotation_quaternion_y = np.array([np.cos(angle_y / 2), 0, np.sin(angle_y / 2), 0])
        rotation_quaternion_z = np.array([np.cos(angle_z / 2), 0, 0, np.sin(angle_z / 2)])

        # Rotate each quaternion in the set
        rotated_quaternions = []
        for i in range(len(quaternion_columns)):
            quaternion = quaternion_columns.iloc[i].to_numpy()
            rotated_quaternion_x = quaternion_multiply(rotation_quaternion_x, quaternion)
            rotated_quaternion_xy = quaternion_multiply(rotation_quaternion_y, rotated_quaternion_x)
            rotated_quaternion_xyz = quaternion_multiply(rotation_quaternion_z, rotated_quaternion_xy)
            rotated_quaternions.append(rotated_quaternion_xyz)

        # Update the corresponding columns in the DataFrame with the rotated quaternions
        for i, col in enumerate(quaternion_columns.columns):
            data[col] = [q[i] for q in rotated_quaternions]

        # Save the updated DataFrame to the output CSV file
        data.to_csv(data_file_path.replace('.csv', '_rotated.csv'), index=False, header=False)

def interpolate_sensor_data(original_data_dir, file_names):
    """
    Interpolates missing timesteps in sensor data files.

    Args:
    - original_data_dir (str): Directory containing the original sensor data files.
    - file_names (list of str): List of file names of the sensor data files to be interpolated.
    """

    for file in file_names:
        # Read the sensor data from the CSV file
        file_path = os.path.join(original_data_dir, file)
        data = pd.read_csv(file_path, header=None)

        # Interpolate missing timesteps
        new_data = []
        for i in range(len(data) - 1):
            new_data.append(data.iloc[i].tolist())  # Append individual row
            current_time = round(data.iloc[i, 0], 2)
            next_time = round(data.iloc[i + 1, 0], 2)
            gap = round(next_time - current_time, 2)
            if gap > 0.01:  # Check if there is a jump
                num_missing_steps = int(gap / 0.01) - 1
                step_size = (data.iloc[i + 1] - data.iloc[i]) / (num_missing_steps + 1)
                for j in range(1, num_missing_steps + 1):
                    interpolated_step = (data.iloc[i] + step_size * j).tolist()  # Convert to list
                    quaternion_values = np.array(interpolated_step[-4:])
                    norm = np.linalg.norm(quaternion_values)
                    normalized_quaternion = quaternion_values / norm
                    interpolated_step[-4:] = normalized_quaternion.tolist()  # Convert to list
                    new_data.append(interpolated_step)  # Append list of interpolated row

        new_data.append(data.iloc[-1].tolist())  # Add the last row as a list

        # Convert new data to DataFrame
        interpolated_df = pd.DataFrame(new_data)

        interpolated_df[0] = interpolated_df[0].round(2)

        # Save interpolated data to CSV file
        interpolated_file_path = os.path.join(original_data_dir, file)
        interpolated_df.to_csv(interpolated_file_path, index=False, header=False)

def synchronize_csv_files(data_dir, csv_files):
    """
    Synchronize the CSV files located in the data directory.

    Args:
    - data_dir (str): Directory containing the original CSV files.
    - csv_files (list of str): List of CSV file names to be processed.
    """

    last_values = []

    # Read the last value from the first column of each CSV file and create new files
    for csv_file in csv_files:
        input_file_path = os.path.join(data_dir, csv_file)
        output_file_path = os.path.join(data_dir, csv_file.replace('.csv', '_sync.csv'))
        
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
    last_values = [round(x * 100) for x in last_values]

    # Find the lowest value among the last values
    min_value = min(last_values)

    # Calculate the differences between each last value and the lowest value
    differences = [value - min_value for value in last_values]

    # Process each CSV file
    for index, csv_file in enumerate(csv_files):
        input_file_path = os.path.join(data_dir, csv_file)
        output_file_path = os.path.join(data_dir, csv_file.replace('.csv', '_sync.csv'))
        num_rows_to_remove = differences[index]  # Number of rows to remove from the beginning

        with open(input_file_path, mode='r') as input_file, open(output_file_path, mode='w', newline='') as output_file:
            reader = csv.reader(input_file)
            writer = csv.writer(output_file)

            # Skip the specified number of rows
            for _ in range(num_rows_to_remove):
                next(reader)

            # Rewrite the values in the first column starting from 0.01 and increasing by 0.01 at every row
            current_time = 0.01
            for row in reader:
                row[0] = str(round(current_time, 2))
                writer.writerow(row)
                current_time += 0.01

    # Indices of columns to be removed (0-indexed)
    columns_to_remove = [4, 5, 6, 10, 11, 12]
    
    # Remove specified columns from the newly created CSV files
    for csv_file in csv_files:
        output_file_path = os.path.join(data_dir, csv_file.replace('.csv', '_sync.csv'))
        
        with open(output_file_path, mode='r') as file:
            lines = file.readlines()
            
        with open(output_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # Write rows back to the file, removing specified columns
            for line in lines:
                values = line.strip().split(',')
                updated_values = [value for index, value in enumerate(values) if index not in columns_to_remove]
                writer.writerow(updated_values)

def merge_csv_files(data_dir, output_file_name, csv_files):
    """
    Merge synchronized CSV files into a single file where each row contains the respective rows from all files.

    Args:
    - data_dir (str): Directory containing the synchronized CSV files.
    - output_file_name (str): Name of the output merged CSV file.
    - csv_files (list of str): List of synchronized CSV file names to be merged.
    """

    # Create a list to hold file handles for input files
    input_file_handles = []

    # Open input files and create file handles
    for csv_file in csv_files:
        input_file_path = os.path.join(data_dir, csv_file.replace('.csv', '_sync.csv'))
        input_file_handles.append(open(input_file_path, mode='r'))

    # Create the output file
    output_file_path = os.path.join(data_dir, output_file_name)
    with open(output_file_path, mode='w', newline='') as output_file:
        writer = csv.writer(output_file)

        # Merge rows from all input files
        while True:
            rows = []
            end_of_files = False

            # Read one row from each input file
            for file_handle in input_file_handles:
                row = next(csv.reader(file_handle), None)
                if row is None:
                    end_of_files = True
                    break
                rows.append(row)

            # Check if all input files reached EOF
            if end_of_files:
                break

            # Write the merged row to the output file
            writer.writerow([item for sublist in rows for item in sublist])

    # Close all input file handles
    for file_handle in input_file_handles:
        file_handle.close()

    merged_file_path = f'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/P{person}/W{weight}/A{attempt}/imu/merged_data.csv'
    
    return merged_file_path

def create_opensim_file(data_dir, merged_file):
    # File paths
    quaternion_table_path = os.path.join(data_dir, 'quaternion_table.csv')
    output_file_path_to_dataset = os.path.join(data_dir, 'lifting_orientations.sto')
    output_file_path_to_opensim = 'c:/Users/giaco/Documents/OpenSim/4.5/Code/Python/OpenSenseExample/lifting_orientations.sto'
    output_file_path_to_opensense = 'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/OpenSense/lifting_orientations.sto'

    # Indices of columns to keep
    columns_to_keep = [0, 10, 11, 12, 13, 24, 25, 26, 27, 38, 39, 40, 41, 52, 53, 54, 55]

    # Read data from the input file and write filtered data to the output file
    with open(merged_file, mode='r') as input_file, open(quaternion_table_path, mode='w', newline='') as output_file:
        reader = csv.reader(input_file)
        writer = csv.writer(output_file)

        for row in reader:
            filtered_row = [row[i] for i in columns_to_keep]
            writer.writerow(filtered_row)

    # Read the filtered data from the input file
    with open(quaternion_table_path, mode='r') as file:
        lines = file.readlines()

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

    # Write the modified data to the output files
    for output_path in [output_file_path_to_dataset, output_file_path_to_opensim, output_file_path_to_opensense]:
        with open(output_path, mode='w') as file:
            file.write(header)
            file.writelines(data_lines)

def delete_files(data_dir, files_to_delete):
    for file_name in files_to_delete:
        file_path = os.path.join(data_dir, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)

def opensim_processing(show_placer=False, show_ik=False):
    # Setup and run the IMUPlacer tool, with model visualization set to true
    imu_placer = osim.IMUPlacer('c:/Users/giaco/Documents/OpenSim/4.5/Code/Python/OpenSenseExample/myIMUPlacer_Setup.xml')
    imu_placer.run(show_placer)

    # Write the calibrated model to file
    calibrated_model = imu_placer.getCalibratedModel()

    # Setup and run the IMU IK tool with visualization set to true
    imu_ik = osim.IMUInverseKinematicsTool('c:/Users/giaco/Documents/OpenSim/4.5/Code/Python/OpenSenseExample/myIMUIK_Setup.xml')
    imu_ik.run(show_ik)

def process_motion_data(motion_file_path, fs, cutoff_frequency_pos=5, cutoff_frequency_vel=5, cutoff_frequency_acc=5):
    # Read the motion data file into a DataFrame
    motion_data = pd.read_csv(motion_file_path, delimiter='\t', skiprows=6)

    def calculate_angular_velocity(angles, timestamps):
        velocities = [0]
        for i in range(1, len(angles)):
            delta_angle = angles[i] - angles[i-1]
            delta_time = timestamps[i] - timestamps[i-1]
            velocity = delta_angle / delta_time
            velocities.append(velocity)
        return velocities

    def calculate_angular_acceleration(velocities, timestamps):
        accelerations = [0]
        for i in range(1, len(velocities)):
            delta_velocity = velocities[i] - velocities[i-1]
            delta_time = timestamps[i] - timestamps[i-1]
            acceleration = delta_velocity / delta_time
            accelerations.append(acceleration)
        return accelerations

    def butter_lowpass(cutoff, fs, order=5):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def apply_lowpass_filter(data, cutoff_frequency, sampling_frequency, filter_order=5):
        b, a = butter_lowpass(cutoff_frequency, sampling_frequency, order=filter_order)
        filtered_data = filtfilt(b, a, data)
        return filtered_data
    
    time = motion_data['time']
    shoulder_flex_angle = motion_data['arm_flex_r']
    shoulder_add_angle = motion_data['arm_add_r']
    elbow_flex_angle = motion_data['elbow_flex_r']
    lumbar_angle = motion_data['lumbar_extension']

    sampling_frequency = fs

    elbow_flex_angle_filt = apply_lowpass_filter(elbow_flex_angle, cutoff_frequency_pos, sampling_frequency)
    shoulder_flex_angle_filt = apply_lowpass_filter(shoulder_flex_angle, cutoff_frequency_pos, sampling_frequency)
    shoulder_add_angle_filt = apply_lowpass_filter(shoulder_add_angle, cutoff_frequency_pos, sampling_frequency)
    lumbar_angle_filt = apply_lowpass_filter(lumbar_angle, cutoff_frequency_pos, sampling_frequency)

    shoulder_flex_vel = calculate_angular_velocity(shoulder_flex_angle_filt, time)
    shoulder_add_vel = calculate_angular_velocity(shoulder_add_angle_filt, time)
    elbow_flex_vel = calculate_angular_velocity(elbow_flex_angle_filt, time)
    lumbar_vel = calculate_angular_velocity(lumbar_angle_filt, time)

    elbow_flex_vel_filt = apply_lowpass_filter(elbow_flex_vel, cutoff_frequency_vel, sampling_frequency)
    shoulder_flex_vel_filt = apply_lowpass_filter(shoulder_flex_vel, cutoff_frequency_vel, sampling_frequency)
    shoulder_add_vel_filt = apply_lowpass_filter(shoulder_add_vel, cutoff_frequency_vel, sampling_frequency)
    lumbar_vel_filt = apply_lowpass_filter(lumbar_vel, cutoff_frequency_vel, sampling_frequency)

    shoulder_flex_acc = calculate_angular_acceleration(shoulder_flex_vel_filt, time)
    shoulder_add_acc = calculate_angular_acceleration(shoulder_add_vel_filt, time)
    elbow_flex_acc = calculate_angular_acceleration(elbow_flex_vel_filt, time)
    lumbar_acc = calculate_angular_acceleration(lumbar_vel_filt, time)

    elbow_flex_acc_filt = apply_lowpass_filter(elbow_flex_acc, cutoff_frequency_acc, sampling_frequency)
    shoulder_flex_acc_filt = apply_lowpass_filter(shoulder_flex_acc, cutoff_frequency_acc, sampling_frequency)
    shoulder_add_acc_filt = apply_lowpass_filter(shoulder_add_acc, cutoff_frequency_acc, sampling_frequency)
    lumbar_acc_filt = apply_lowpass_filter(lumbar_acc, cutoff_frequency_acc, sampling_frequency)

    return {
        'time': time,
        'shoulder_flex_angle_filt': shoulder_flex_angle_filt,
        'shoulder_add_angle_filt': shoulder_add_angle_filt,
        'elbow_flex_angle_filt': elbow_flex_angle_filt,
        'lumbar_angle_filt': lumbar_angle_filt,
        'shoulder_flex_vel_filt': shoulder_flex_vel_filt,
        'shoulder_add_vel_filt': shoulder_add_vel_filt,
        'elbow_flex_vel_filt': elbow_flex_vel_filt,
        'lumbar_vel_filt': lumbar_vel_filt,
        'shoulder_flex_acc_filt': shoulder_flex_acc_filt,
        'shoulder_add_acc_filt': shoulder_add_acc_filt,
        'elbow_flex_acc_filt': elbow_flex_acc_filt,
        'lumbar_acc_filt': lumbar_acc_filt
    }

def save_plot_motion_data(data_dir, motion_data, plot_name, show_plot=False):
    # Define the figure and subplots
    fig, axs = plt.subplots(3, 4, figsize=(15, 15))

    # Plot the filtered elbow angle
    axs[0, 0].plot(motion_data['time'], motion_data['elbow_flex_angle_filt'], color='blue')
    axs[0, 0].set_ylabel('Elbow Angle (degrees)')
    axs[0, 0].set_title('Elbow Angle')
    axs[0, 0].grid(True)

    # Plot the filtered shoulder flexion angle
    axs[0, 1].plot(motion_data['time'], motion_data['shoulder_flex_angle_filt'], color='green')
    axs[0, 1].set_ylabel('Shoulder Flexion Angle (degrees)')
    axs[0, 1].set_title('Shoulder Flexion Angle')
    axs[0, 1].grid(True)

    # Plot the filtered shoulder adduction angle
    axs[0, 2].plot(motion_data['time'], motion_data['shoulder_add_angle_filt'], color='red')
    axs[0, 2].set_ylabel('Shoulder Adduction Angle (degrees)')
    axs[0, 2].set_title('Shoulder Adduction Angle')
    axs[0, 2].grid(True)

    # Plot the filtered shoulder adduction angle
    axs[0, 3].plot(motion_data['time'], motion_data['lumbar_angle_filt'], color='black')
    axs[0, 3].set_ylabel('Lumbar Angle (degrees)')
    axs[0, 3].set_title('Lumbar Angle')
    axs[0, 3].grid(True)

    # Plot the filtered elbow velocity
    axs[1, 0].plot(motion_data['time'], motion_data['elbow_flex_vel_filt'], color='blue')
    axs[1, 0].set_ylabel('Elbow Velocity (degrees/s)')
    axs[1, 0].set_title('Elbow Velocity')
    axs[1, 0].grid(True)

    # Plot the filtered shoulder flexion velocity
    axs[1, 1].plot(motion_data['time'], motion_data['shoulder_flex_vel_filt'], color='green')
    axs[1, 1].set_ylabel('Shoulder Flexion Velocity (degrees/s)')
    axs[1, 1].set_title('Shoulder Flexion Velocity')
    axs[1, 1].grid(True)

    # Plot the filtered shoulder adduction velocity
    axs[1, 2].plot(motion_data['time'], motion_data['shoulder_add_vel_filt'], color='red')
    axs[1, 2].set_ylabel('Shoulder Adduction Velocity (degrees/s)')
    axs[1, 2].set_title('Shoulder Adduction Velocity')
    axs[1, 2].grid(True)

    # Plot the filtered shoulder adduction angle
    axs[1, 3].plot(motion_data['time'], motion_data['lumbar_vel_filt'], color='black')
    axs[1, 3].set_ylabel('Lumbar Velocity (degrees/s)')
    axs[1, 3].set_title('Lumbar Velocity')
    axs[1, 3].grid(True)

    # Plot the filtered elbow acceleration
    axs[2, 0].plot(motion_data['time'], motion_data['elbow_flex_acc_filt'], color='blue')
    axs[2, 0].set_ylabel('Elbow Acceleration (degrees/s^2)')
    axs[2, 0].set_title('Elbow Acceleration')
    axs[2, 0].grid(True)

    # Plot the filtered shoulder flexion acceleration
    axs[2, 1].plot(motion_data['time'], motion_data['shoulder_flex_acc_filt'], color='green')
    axs[2, 1].set_ylabel('Shoulder Flexion Acceleration (degrees/s^2)')
    axs[2, 1].set_title('Shoulder Flexion Acceleration')
    axs[2, 1].grid(True)

    # Plot the filtered shoulder adduction acceleration
    axs[2, 2].plot(motion_data['time'], motion_data['shoulder_add_acc_filt'], color='red')
    axs[2, 2].set_ylabel('Shoulder Adduction Acceleration (degrees/s^2)')
    axs[2, 2].set_title('Shoulder Adduction Acceleration')
    axs[2, 2].grid(True)

    # Plot the filtered shoulder adduction angle
    axs[2, 3].plot(motion_data['time'], motion_data['lumbar_acc_filt'], color='black')
    axs[2, 3].set_ylabel('Lumbar Acceleration (degrees/s^2)')
    axs[2, 3].set_title('Lumbar Acceleration')
    axs[2, 3].grid(True)

    # Set common xlabel
    for ax in axs.flatten():
        ax.set_xlabel('Time')

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(data_dir, plot_name))

    # # Show the plots
    if show_plot:
        plt.show()

def save_motion_data(data_dir, motion_data, file_name):
    # Define the file path for saving the CSV file
    csv_file_path = os.path.join(data_dir, file_name)

    # Write the data to the CSV file
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header
        writer.writerow(motion_data.keys())
        
        # Write the data rows
        for values in zip(*motion_data.values()):
            writer.writerow(values)




for person in range(1, tot_person + 1):
    for weight in range(1, tot_weights + 1):
        for attempt in range(1, tot_attempts + 1):

            # Directory where CSV files are located for the current person, weight, and attempt
            data_dir = os.path.join(base_dir, f'P{person}/W{weight}/A{attempt}/imu')
            file_names = ['sensor1.csv', 'sensor2.csv', 'sensor3.csv', 'sensor4.csv']
            csv_files = ['sensor1_rotated.csv', 'sensor2_rotated.csv', 'sensor3_rotated.csv', 'sensor4_rotated.csv']

            # Apply a rotation of 270° to the pelvis sensor to have it facing forward in OpenSim
            angle_x = np.pi
            angle_y = np.pi
            angle_z = np.pi
            rotate_up_low_arm(data_dir, 0, angle_y*1/2, 0, file_names)

            rotate_pelvis_torso(data_dir, 0, angle_y, 0, file_names)

            # Interpolate the missing values in the sensor readings
            interpolate_sensor_data(data_dir, csv_files)

            # Align all the data of the sensors that started recording at different times
            synchronize_csv_files(data_dir, csv_files)

            # Merge the data from the sensors into a single file
            merged_file = merge_csv_files(data_dir, 'merged_data.csv', csv_files)

            # Create the .sto file to use with OpenSim
            create_opensim_file(data_dir, merged_file)

            # List of files to delete
            files_to_delete = [
                'sensor1_rotated.csv',
                'sensor1_rotated_sync.csv',
                'sensor2_rotated.csv',
                'sensor2_rotated_sync.csv',
                'sensor3_rotated.csv',
                'sensor3_rotated_sync.csv',
                'sensor4_rotated.csv',
                'sensor4_rotated_sync.csv',
                'quaternion_table.csv'
            ]

            # Delete the files that are not useful anymore
            delete_files(data_dir, files_to_delete)

            # Perform the inverse kinematics through OpenSim
            opensim_processing(False, False)

            print(f'Data processing complete for Person {person}/{tot_person}, Weight {weight}/{tot_weights}, Attempt {attempt}/{tot_attempts}')

            # Process the data obtained with OpenSim to get filtered angles, velocities, and accelerations
            motion_data = 'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/OpenSense/IKResults/ik_lifting_orientations.mot'
            motion_data_processed = process_motion_data(motion_data, 100, 5, 5, 5)

            # Save the plots and the data of the joints
            save_plot_motion_data(data_dir, motion_data_processed, 'joint_data.png', False)
            save_motion_data(data_dir, motion_data_processed, 'data_neural.csv')

print('Done')