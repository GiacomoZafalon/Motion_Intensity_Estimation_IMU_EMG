import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import numpy as np
import os
import shutil

def smooth_euler_angles(angles):
    angles_1 = angles[:, 0]
    angles_3 = angles[:, 2]
    jump = 0
    jump_3 = 0
    diff_1_prec = 0
    diff_3_prec = 0
    for i in range(1, len(angles)):
        diff_1 = angles_1[i] - angles_1[i - 1]
        angles_1[i-1] = angles_1[i-1] - jump
        diff_3 = angles_3[i] - angles_3[i - 1]
        angles_3[i-1] = angles_3[i-1] - jump_3
        if abs(diff_1) > 300:
            angles_1[i] = angles_1[i] - (360 * diff_1/abs(diff_1))
        if abs(diff_1) > 5 and abs(diff_1) < 300:
            if abs(diff_1_prec - diff_1) > 5:
                jump += diff_1
            diff_1_prec = diff_1
        if abs(diff_3) > 5:
            if abs(diff_3_prec - diff_3) > 5:
                jump_3 += diff_3
            diff_3_prec = diff_3
    angles_1[-1] = angles_1[-1] - jump
    angles_3[-1] = angles_3[-1] - jump_3
    return angles_1, angles_3

def interpolate_missing_data(df):
    cols_to_interpolate = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    df.iloc[:, cols_to_interpolate] = df.iloc[:, cols_to_interpolate].replace(0, np.nan)
    df.iloc[:, cols_to_interpolate] = df.iloc[:, cols_to_interpolate].interpolate(method='linear').round(2).ffill().bfill()
    return df

def process_quaternions(directory, filenames):
    for file in filenames:
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path)

        df = interpolate_missing_data(df)
        
        euler_angles = df.iloc[:, 1:4].values
        angles_1, angles_3 = smooth_euler_angles(euler_angles)

        df.iloc[:, 1] = -angles_1
        df.iloc[:, 3] = angles_3
        euler_angles = df.iloc[:, 1:4].values

        r = R.from_euler('zyx', euler_angles, degrees=True)
        quaternions = r.as_quat()

        quat_df = pd.DataFrame(quaternions, columns=['qx', 'qy', 'qz', 'qw'])
        df.iloc[:, -4:] = quat_df
        
        df.to_csv(file_path.replace('.csv', '_rot_quat.csv'), index=False, header=False)

def remove_columns_from_csv_files(data_dir, csv_files, columns_to_remove):
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(data_dir, csv_file))
        df.drop(columns=df.columns[columns_to_remove], inplace=True)
        df.to_csv(os.path.join(data_dir, f"{csv_file.replace('.csv', '_processed.csv')}"), index=False)

def merge_csv_files_reverse_side_by_side_append_top(data_dir, csv_files, output_file_name):
    columns_to_remove = [4, 5, 6, 10, 11, 12]
    remove_columns_from_csv_files(data_dir, csv_files, columns_to_remove)

    data_frames = [pd.read_csv(os.path.join(data_dir, f"{csv_file.replace('.csv', '_processed.csv')}")).iloc[::-1].reset_index(drop=True) for csv_file in csv_files]
    max_length = max(len(df) for df in data_frames)
    merged_rows = []

    for i in range(max_length):
        merged_row = []
        for df in data_frames:
            if i < len(df):
                merged_row.extend(df.iloc[i].values)
            else:
                merged_row.extend([np.nan] * df.shape[1])
        merged_rows.insert(0, merged_row)

    merged_data = pd.DataFrame(merged_rows)
    merged_data.drop(columns=[14, 28, 42], inplace=True, errors='ignore')

    output_file_path = os.path.join(data_dir, output_file_name)
    merged_data.to_csv(output_file_path, index=False, header=False)

    # Delete intermediate processed files
    for csv_file in csv_files:
        processed_file_path = os.path.join(data_dir, f"{csv_file.replace('.csv', '_processed.csv')}")
        if os.path.exists(processed_file_path):
            os.remove(processed_file_path)

    return output_file_path

def remove_empty_top_rows_and_adjust(file_path, num_files):
    df = pd.read_csv(file_path, header=None)
    cols_per_file = df.shape[1] // num_files

    for i in range(len(df)):
        row = df.iloc[i]
        if all([not row[j:j+cols_per_file].isna().all() for j in range(0, len(row), cols_per_file)]):
            break

    # Remove empty top rows
    df = df.iloc[i:]

    # Adjust the first column to start from 0.01 and increment by 0.01
    df.iloc[:, 0] = np.arange(0.01, (len(df) * 0.01) + 0.01, 0.01)

    df.to_csv(file_path, index=False, header=False)

def process_merged_data(input_file_path, output_file_path):
    # Read the merged data file
    df = pd.read_csv(input_file_path, header=None)

    # Find the index of the row with the maximum value in column 28
    max_value_index = df[28].idxmax()

    # Skip 10 rows from the maximum value index
    start_index = max_value_index + 10

    # Slice the DataFrame to keep rows from start_index onwards
    df_final = df.iloc[:start_index]

    # Save the final DataFrame to a new CSV file
    df_final.to_csv(output_file_path, index=False, header=False)

    return df_final

def plot_column_groups(df_final):
    # Define the groups of columns to plot
    column_groups = [
        [1, 14, 27, 40],
        [2, 15, 28, 41],
        [3, 16, 29, 42]
    ]

    # Create subplots for each group
    fig, axes = plt.subplots(len(column_groups), 1, figsize=(10, 12), sharex=True)

    # Plot each group of columns
    for i, group in enumerate(column_groups):
        for col_idx in group:
            axes[i].plot(df_final.index, df_final[col_idx], label=f'Column {col_idx}')
        axes[i].set_title(f'Group {i+1}')
        axes[i].set_ylabel('Value')
        axes[i].legend()

    # Set common xlabel
    axes[-1].set_xlabel('Index')

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

tot_persons = 1
tot_weights = 1
tot_attempts = 1


for person in range(1, tot_persons + 1):
    for weight in range(1, tot_weights + 1):
        for attempt in range(1, tot_attempts + 1):
            data_dir = rf'C:\Users\giaco\OneDrive\Desktop\Università\Tesi_Master\GitHub\Dataset\P{person}\W{weight}\A{attempt}\imu'
            file_names = ['sensor1.csv', 'sensor2.csv', 'sensor3.csv', 'sensor4.csv']
            csv_files = ['sensor1_rot_quat.csv', 'sensor2_rot_quat.csv', 'sensor3_rot_quat.csv', 'sensor4_rot_quat.csv']
            output_file_name = 'merged_data.csv'

            process_quaternions(data_dir, file_names)
            merged_file_path = merge_csv_files_reverse_side_by_side_append_top(data_dir, csv_files, output_file_name)
            remove_empty_top_rows_and_adjust(merged_file_path, len(csv_files))

            for csv_file in csv_files:
                quaternion_file_path = os.path.join(data_dir, f"{csv_file.replace('.csv', '_rot_quat.csv')}")
                if os.path.exists(quaternion_file_path):
                    os.remove(quaternion_file_path)
            for csv_file in file_names:
                quaternion_file_path = os.path.join(data_dir, f"{csv_file.replace('.csv', '_rot_quat.csv')}")
                if os.path.exists(quaternion_file_path):
                    os.remove(quaternion_file_path)

            input_file_path = os.path.join(data_dir, 'merged_data.csv')
            output_file_path = os.path.join(data_dir, 'merged_file_final.csv')

            df_final = process_merged_data(input_file_path, output_file_path)
            plot_column_groups(df_final)