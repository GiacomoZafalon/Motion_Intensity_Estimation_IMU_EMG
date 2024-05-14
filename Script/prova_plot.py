import numpy as np
import os

def rotate_quaternions_in_files(directory, filenames):
    """
    Rotate quaternions in each CSV file in the specified directory using the
    first quaternion as the target quaternion and save the modified data to new files.

    Args:
    - directory (str): Path to the directory containing the CSV files.
    - filenames (list): List of CSV filenames to process.
    """
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

    def quaternion_to_rotation_matrix(q):
        w, x, y, z = q
        rotation_matrix = np.array([[1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
                                    [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
                                    [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]])
        return rotation_matrix

    def rotation_matrix_to_quaternion(R):
        tr = R[0, 0] + R[1, 1] + R[2, 2]
        
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
        
        return np.array([qw, qx, qy, qz])


    def rotate_to_target(q, q0):
        # First rotation: Align the quaternion with [1, 0, 0, 0]
        target_quaternion = np.array([1, 0, 0, 0])
        rotation_quaternion = quaternion_multiply(target_quaternion, quaternion_conjugate(q0))
        rotated_q = quaternion_multiply(rotation_quaternion, q)
        
        # Second rotation: Rotate the quaternion to align with the new reference frame
        # Define the rotation matrix from the old reference frame to the new reference frame
        rotation_matrix = np.array([[0, 0, 1],
                                    [0, 1, 0],
                                    [-1, 0, 0]])
        
        # Convert the quaternion to a rotation matrix
        rotation_matrix_original = quaternion_to_rotation_matrix(rotated_q)
        
        # Apply the rotation to align the old reference frame with the new reference frame
        rotated_rotation_matrix = np.dot(rotation_matrix_original, rotation_matrix)
        
        # Convert the rotated rotation matrix back to a quaternion
        rotated_q = rotation_matrix_to_quaternion(rotated_rotation_matrix)

        return rotated_q

    # Loop through each CSV file
    for filename in filenames:
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


base_dir = 'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/'
data_dir = os.path.join(base_dir, f'P1/W1/A1/imu')
file_names = ['sensor1.csv', 'sensor2.csv', 'sensor3.csv', 'sensor4.csv']


# Example usage
rotate_quaternions_in_files(data_dir, file_names)