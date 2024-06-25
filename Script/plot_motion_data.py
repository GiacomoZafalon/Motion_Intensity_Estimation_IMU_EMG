import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.signal import butter, filtfilt

def plot_euler(df_final, person, weight, attempt):
    # Define the groups of columns to plot
    column_groups = [
        [1, 14, 27, 40],
        [2, 15, 29, 42],
        [3, 16, 28, 41]
    ]

    # Create subplots for each group
    fig, axes = plt.subplots(len(column_groups), 1, figsize=(10, 12), sharex=True)

    # Plot each group of columns
    for i, group in enumerate(column_groups):
        for col_idx in group:
            axes[i].plot(df_final.index, df_final[col_idx], label=f'Column {col_idx}')
        axes[i].set_title(f'[{person}, {weight}, {attempt}]')
        axes[i].set_ylabel('Value')
        if i == 0:
            axes[i].set_ylim(-200, 0)
        elif i == 1:
            axes[i].set_ylim(-20, 180)
        elif i == 2:
            axes[i].set_ylim(-20, 360)
        axes[i].legend()

    # Set common xlabel
    axes[-1].set_xlabel('Index')

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()


def plot_accelerations(df_final, person, weight, attempt):
    # Define the groups of columns to plot
    column_groups = [
        [4, 17, 30, 43],
        [5, 18, 32, 45],
        [6, 19, 31, 44]
    ]

    # Create subplots for each group
    fig, axes = plt.subplots(len(column_groups), 1, figsize=(10, 12), sharex=True)

    # Plot each group of columns
    for i, group in enumerate(column_groups):
        for col_idx in group:
            axes[i].plot(df_final.index, df_final[col_idx], label=f'Column {col_idx}')
        axes[i].set_title(f'[{person}, {weight}, {attempt}]')
        axes[i].set_ylabel('Value')
        axes[i].set_ylim(0, 4.0)
        axes[i].legend()

    # Set common xlabel
    axes[-1].set_xlabel('Index')

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

def plot_quaternions(sto_file, person, weight, attempt):
    # Read the STO file, skipping the header
    with open(sto_file, 'r') as f:
        lines = f.readlines()
    
    # Find the line where the actual data starts
    data_start_idx = 0
    for idx, line in enumerate(lines):
        if line.startswith('endheader'):
            data_start_idx = idx + 1
            break
    
    # Read the data into a DataFrame
    df = pd.read_csv(sto_file, delimiter='\t', skiprows=data_start_idx)
    
    # Extract quaternion columns
    columns = ['pelvis_imu', 'torso_imu', 'humerus_r_imu', 'ulna_r_imu']
    quaternion_data = {}
    
    for col in columns:
        if col in df.columns:
            # Split the column into separate quaternion components
            quat_components = df[col].str.split(',', expand=True).astype(float)
            quat_components.columns = [f'{col}_w', f'{col}_x', f'{col}_y', f'{col}_z']
            quaternion_data[col] = quat_components
        else:
            print(f"Error: {col} does not have 4 quaternion components.")

    # Plot each set of quaternions
    fig, axs = plt.subplots(len(columns), 1, figsize=(10, 20))
    time = df['time']
    for idx, (col, quaternions) in enumerate(quaternion_data.items()):
        if quaternions.shape[1] == 4:
            axs[idx].plot(time, quaternions[f'{col}_w'], label=f'{col}_w')
            axs[idx].plot(time, quaternions[f'{col}_x'], label=f'{col}_x')
            axs[idx].plot(time, quaternions[f'{col}_y'], label=f'{col}_y')
            axs[idx].plot(time, quaternions[f'{col}_z'], label=f'{col}_z')
            axs[idx].set_title(f'[{person}, {weight}, {attempt}]')
            axs[idx].legend()
            axs[idx].set_xlabel('Time')
            axs[idx].set_ylabel('Quaternion value')
        else:
            print(f"Error: {col} data does not have 4 quaternion components.")
    
    plt.tight_layout()
    plt.show()

# Example usage:
person = 13
weight = 5
attempt = 1
data_dir = rf'C:\Users\giaco\OneDrive\Desktop\Università\Tesi_Master\GitHub\Dataset\P{person}\W{weight}\A{attempt}\imu'
merged_file_path = os.path.join(data_dir, 'merged_file_final_filt.csv')
sto_file_path = os.path.join(data_dir, 'lifting_orientations.sto')

# Plotting the merged data file
df_filt = pd.read_csv(merged_file_path, header=None)
# plot_euler(df_final, person, weight, attempt)

plot_accelerations(df_filt, person, weight, attempt)

# Plotting the quaternion data from the .sto file
# plot_quaternions(sto_file_path, person, weight, attempt)
