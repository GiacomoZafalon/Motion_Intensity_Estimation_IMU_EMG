import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import numpy as np
import os

def smooth_euler_angles(angles):
    angles_1 = angles[:, 0]
    angles_3 = angles[:, 2]
    for i in range(1, len(angles)):
        diff_1 = angles_1[i] - angles_1[i - 1]
        diff_3 = angles_3[i] - angles_3[i - 1]
        if abs(diff_1) > 300:
            # print('here', diff_1, diff_1/abs(diff_1))
            angles_1[i] = angles_1[i] - (360 * diff_1/abs(diff_1))
        if abs(diff_1) > 100 and abs(diff_1) < 300:
            angles_1[i] = angles_1[i] - diff_1
        if abs(diff_3) > 100:
            angles_3[i] = angles_3[i] - (180 * diff_3/abs(diff_3))
    return angles_1, angles_3

def interpolate_missing_data(df):
    cols_to_interpolate = [1, 2, 3]
    df.iloc[:, cols_to_interpolate] = df.iloc[:, cols_to_interpolate].replace(0, np.nan)  # Replace zeros with NaN for interpolation
    df.iloc[:, cols_to_interpolate] = df.iloc[:, cols_to_interpolate].interpolate(method='linear')  # Interpolate missing values
    df.iloc[:, cols_to_interpolate] = df.iloc[:, cols_to_interpolate].fillna(method='bfill')  # Backfill any remaining NaNs
    df.iloc[:, cols_to_interpolate] = df.iloc[:, cols_to_interpolate].fillna(method='ffill')  # Forward fill any remaining NaNs
    return df

def process_quaternions(directory, filenames):
    for file in filenames:
        file_path = os.path.join(directory, file)
        # Load the CSV file
        df = pd.read_csv(file_path)

        df = interpolate_missing_data(df)
        
        # Extract the first 3 columns (Euler angles: roll, pitch, yaw)
        euler_angles = df.iloc[:, 1:4].values
        old_quat = df.iloc[:, -4:].copy().values

        angles_1, angles_3 = smooth_euler_angles(euler_angles)

        df.iloc[:, 1] = angles_1
        df.iloc[:, 3] = angles_3
        euler_angles = df.iloc[:, 1:4].values

        # Convert Euler angles to quaternions
        r = R.from_euler('zyx', euler_angles, degrees=True)
        quaternions = r.as_quat()  # This gives quaternions in the order (x, y, z, w)
        
        # Create a DataFrame for the quaternions
        quat_df = pd.DataFrame(quaternions, columns=['qx', 'qy', 'qz', 'qw'])

        # Negate specific components to match plotting conventions
        # quat_df['qy'] = -quat_df['qy']
        # quat_df['qw'] = -quat_df['qw']
        
        # Replace the last 4 columns with the new quaternions
        df.iloc[:, -4:] = quat_df
        
        # Save the modified DataFrame back to CSV
        df.to_csv(file_path.replace('.csv', '_rot_quat.csv'), index=False, header=False)

        # Create a single figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

        # Plot the original quaternion values
        ax1.plot(old_quat[:, 0], label='qw_old')
        ax1.plot(old_quat[:, 1], label='qx_old')
        ax1.plot(old_quat[:, 2], label='qy_old')
        ax1.plot(old_quat[:, 3], label='qz_old')
        ax1.set_title('Plot of Quaternion Values')
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Quaternion Value')
        
        # Plot the new smoothed quaternion values
        ax2.plot(quat_df['qw'], label='qw_smooth')
        ax2.plot(quat_df['qx'], label='qx_smooth')
        ax2.plot(quat_df['qy'], label='qy_smooth')
        ax2.plot(quat_df['qz'], label='qz_smooth')
        ax2.set_title('Plot of the Smoothed Quaternion Values')
        ax2.set_xlabel('Index')
        ax2.set_ylabel('Quaternion Value')

        ax3.plot(euler_angles[:, 0], label='roll')
        ax3.plot(euler_angles[:, 1], label='pitch')
        ax3.plot(euler_angles[:, 2], label='yaw')
        ax3.set_title('Plot of Euler Angles')
        ax3.set_xlabel('Index')
        ax3.set_ylabel('Euler Angle (degrees)')

        # Adjust layout
        plt.tight_layout()

        # Show the plots
        plt.show()

data_dir = r'C:\Users\giaco\OneDrive\Desktop\Università\Tesi_Master\GitHub\Dataset\P1\W1\A1\imu'
file_names = ['sensor3.csv']
process_quaternions(data_dir, file_names)
