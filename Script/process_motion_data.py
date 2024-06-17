import pandas as pd
import os
import matplotlib.pyplot as plt
import csv
from scipy.signal import butter, filtfilt

def process_motion_data(motion_file_path, fs, cutoff_frequency_pos=5, cutoff_frequency_vel=5, cutoff_frequency_acc=5):
    # Read the motion data file into a DataFrame
    motion_data = pd.read_csv(motion_file_path)

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

    time = motion_data.iloc[:, 0].to_numpy()
    shoulder_flex_angle = motion_data.iloc[:, 1].to_numpy()
    shoulder_add_angle = motion_data.iloc[:, 2].to_numpy()
    elbow_flex_angle = motion_data.iloc[:, 3].to_numpy()
    lumbar_angle = -motion_data.iloc[:, 4].to_numpy()

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
    # Ensure the directory exists
    os.makedirs(data_dir, exist_ok=True)

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

    # Plot the filtered lumbar angle
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

    # Plot the filtered lumbar velocity
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

    # Plot the filtered lumbar acceleration
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

    # Show the plots
    if show_plot:
        plt.show()

def save_motion_data(data_dir, motion_data, file_name):
    # Ensure the directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Define the file path for saving the CSV file
    csv_file_path = os.path.join(data_dir, file_name)

    # Write the data to the CSV file
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # # Write the header
        # writer.writerow(motion_data.keys())
        
        # Write the data rows
        for values in zip(*motion_data.values()):
            writer.writerow(values)

# Define parameters
tot_person = 1
tot_weights = 1
tot_attempts = 1

base_dir = 'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/'

for person in range(1, tot_person + 1):
    for weight in range(1, tot_weights + 1):
        for attempt in range(1, tot_attempts + 1):

            person = 8
            weight = 3
            attempt = 1

            data_dir = os.path.join(base_dir, f'P{person}/W{weight}/A{attempt}/imu')

            # Process the data obtained with OpenSim to get filtered angles, velocities, and accelerations
            motion_data_path = os.path.join(base_dir, f'P{person}/W{weight}/A{attempt}/imu/data_neural.csv')
            motion_data_processed = process_motion_data(motion_data_path, 100, 3, 3, 3)

            # Save the plots and the data of the joints
            save_plot_motion_data(data_dir, motion_data_processed, 'joint_data.png', True)
            save_motion_data(data_dir, motion_data_processed, 'data_neural.csv')

            print(f'Data processing complete for Person {person}/{tot_person}, Weight {weight}/{tot_weights}, Attempt {attempt}/{tot_attempts}')

print('Done')
