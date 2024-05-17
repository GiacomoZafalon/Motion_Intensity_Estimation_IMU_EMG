import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# Load the CSV file
file_path = r'C:\Users\giaco\OneDrive\Desktop\Università\Tesi_Master\GitHub\Dataset\P1\W1\A1\imu\sensor3_rot_quat.csv'  # Adjust the file path if necessary
df = pd.read_csv(file_path)

# Extract the first 3 columns (assumed to be Euler angles: roll, pitch, yaw)
euler_angles = df.iloc[:, 1:4].values

# Convert Euler angles to quaternions
r = R.from_euler('xyz', euler_angles, degrees=True)
quaternions = r.as_quat()  # This gives quaternions in the order (x, y, z, w)

# Convert quaternions to DataFrame for easier plotting
quat_df = pd.DataFrame(quaternions, columns=['qx', 'qy', 'qz', 'qw'])

# Create a single figure with 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))

# Plot the first 3 columns (Euler angles)
ax1.plot(df.iloc[:, 1], label='x')
ax1.plot(df.iloc[:, 2], label='y')
ax1.plot(df.iloc[:, 3], label='z')
ax1.legend()
ax1.set_title('Plot of the first 3 columns (Euler angles)')
ax1.set_xlabel('Index')
ax1.set_ylabel('Value')

# Plot the last 4 columns (original quaternion data)
ax2.plot(df.iloc[:, -4], label='qw')
ax2.plot(df.iloc[:, -3], label='qx')
ax2.plot(df.iloc[:, -2], label='qy')
ax2.plot(df.iloc[:, -1], label='qz')
ax2.legend()
ax2.set_title('Plot of the last 4 columns (original quaternion data)')
ax2.set_xlabel('Index')
ax2.set_ylabel('Value')

# Plot the new quaternions
ax3.plot(quat_df['qw'], label='qw')
ax3.plot(quat_df['qx'], label='qx')
ax3.plot(quat_df['qy'], label='qy')
ax3.plot(quat_df['qz'], label='qz')
ax3.legend()
ax3.set_title('Plot of the new quaternions')
ax3.set_xlabel('Index')
ax3.set_ylabel('Value')

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()
