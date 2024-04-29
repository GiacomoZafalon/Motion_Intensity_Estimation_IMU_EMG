import numpy as np
import pandas as pd
import os

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])

def rotate_quaternions_and_update_csv(input_file_path, output_file_path):
    # Define the rotation quaternion for 180-degree rotation around the x-axis
    rotation_quaternion = np.array([np.sqrt(2)/2, np.sqrt(2)/2, 0, 0])

    # Read the sensor data from the CSV file
    data = pd.read_csv(input_file_path)

    # Extract the quaternion columns (assuming they are the last 4 columns)
    quaternion_columns = data.iloc[:, -4:]

    # Rotate each quaternion in the set
    rotated_quaternions = []
    for i in range(len(quaternion_columns)):
        quaternion = quaternion_columns.iloc[i].to_numpy()
        rotated_quaternion = quaternion_multiply(quaternion, rotation_quaternion)
        rotated_quaternions.append(rotated_quaternion)

    # Update the corresponding columns in the DataFrame with the rotated quaternions
    for i, col in enumerate(quaternion_columns.columns):
        data[col] = [q[i] for q in rotated_quaternions]

    # Save the updated DataFrame to the output CSV file
    data.to_csv(output_file_path, index=False, header=False)

# Example usage:
input_file_path = 'C:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/P1/W1/A1/imu/sensor1.csv'
output_file_path = 'C:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/P1/W1/A1/imu/sensor1_rotated.csv'
rotate_quaternions_and_update_csv(input_file_path, output_file_path)
