import pandas as pd
import os
import matplotlib.pyplot as plt

def plot_motion_data(data_dir, motion_data, plot_name, show_plot=False):
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

def read_motion_data(file_path):
    # Specify the column names manually
    columns = [ 'time',
                'shoulder_flex_angle_filt',
                'shoulder_add_angle_filt',
                'elbow_flex_angle_filt',
                'lumbar_angle_filt',
                'shoulder_flex_vel_filt',
                'shoulder_add_vel_filt',
                'elbow_flex_vel_filt',
                'lumbar_vel_filt',
                'shoulder_flex_acc_filt',
                'shoulder_add_acc_filt',
                'elbow_flex_acc_filt',
                'lumbar_acc_filt']
    return pd.read_csv(file_path, header=None, names=columns)

# Define parameters
tot_person = 1
tot_weights = 1
tot_attempts = 1

base_dir = 'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/'

for person in range(1, tot_person + 1):
    for weight in range(1, tot_weights + 1):
        for attempt in range(1, tot_attempts + 1):

            person = 1
            weight = 1
            attempt = 2

            data_dir = os.path.join(base_dir, f'P{person}/W{weight}/A{attempt}/imu')

            # Read the motion data
            motion_data_path = os.path.join(data_dir, 'data_neural.csv')
            motion_data = read_motion_data(motion_data_path)

            # Plot the data
            plot_motion_data(data_dir, motion_data, 'joint_data.png', True)

            print(f'Data plotting complete for Person {person}/{tot_person}, Weight {weight}/{tot_weights}, Attempt {attempt}/{tot_attempts}')

print('Done')
