import numpy as np
import os

# Define quaternion rotation functions
def normalize_quaternion(q):
    return q / np.linalg.norm(q)

def quaternion_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return np.array([w, x, y, z])

def rotate_to_target(q, q0):
    # Normalize input quaternions
    q0 = normalize_quaternion(q0)
    target = np.array([1, 0, 0, 0])
    target = normalize_quaternion(target)
    
    # Calculate the rotation quaternion that rotates q to the target quaternion
    rotation_quaternion = quaternion_multiply(target, quaternion_conjugate(q0))
    
    # Apply the rotation to q
    rotated_q = quaternion_multiply(rotation_quaternion, q)
    
    return rotated_q

# Directory containing the CSV files
directory = 'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/P1/W1/A1/imu/'

# Loop through each CSV file
for filename in ['sensor1.csv', 'sensor2.csv', 'sensor3.csv', 'sensor4.csv']:
    filepath = os.path.join(directory, filename)
    
    # Load data
    data = np.loadtxt(filepath, delimiter=',')
    
    # Extract quaternion data from the remaining rows
    quaternions = data[:, -4:]
    
    # Rotate the first quaternion to match the target quaternion
    rotated_target_quaternion = rotate_to_target(quaternions[1], quaternions[1])
    
    # Apply the same rotation to all other quaternions
    rotated_quaternions = np.array([rotate_to_target(q, quaternions[1]) for q in quaternions])
    
    # Replace the original quaternion data with the rotated quaternions
    data[:, -4:] = rotated_quaternions
    
    # Save the modified data back to a file
    output_filepath = os.path.join(directory, filename.replace('.csv', '_rot_quat.csv'))
    np.savetxt(output_filepath, data, delimiter=',')
