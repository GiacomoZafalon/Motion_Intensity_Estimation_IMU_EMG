import matplotlib.pyplot as plt
import numpy as np
import os

# Data
weights = ['W1', 'W2', 'W3', 'W4', 'W5']
avg_duration = [1.40, 1.64, 1.76, 1.85, 2.08]
avg_acceleration = [0.42, 0.39, 0.36, 0.37, 0.36]
max_acceleration = [3.83, 3.62, 3.55, 3.67, 3.46]
avg_max_acceleration = [2.78, 2.62, 2.53, 2.53, 2.33]

# Directory to save plots
save_dir = r'C:\Users\giaco\OneDrive\Desktop\Università\Tesi_Master\Results\Kinematics'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Plot 1: Average Duration
plt.figure(figsize=(8, 6))
plt.plot(weights, avg_duration, marker='o', linestyle='-', color='b')
plt.title('Average Duration for Weights')
plt.xlabel('Weights')
plt.ylabel('Average Duration [s]')
plt.ylim(1, 2.5)
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'average_duration.png'))
plt.close()

# Plot 2: Acceleration Data
plt.figure(figsize=(8, 6))
index = np.arange(len(weights))

# Adding line plots
plt.plot(index, max_acceleration, marker='o', linestyle='-', color='r', label='Max Acceleration')
plt.plot(index, avg_max_acceleration, marker='o', linestyle='-', color='y', label='Avg Max Acceleration')

plt.xlabel('Weights')
plt.ylabel('Acceleration [m/s$^2$]')
plt.title('Max and Avg Max Acceleration for Weights')
plt.xticks(index, weights)
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'max_avg_max_acceleration.png'))
plt.close()

# Plot 3: Average Acceleration
plt.figure(figsize=(8, 6))
plt.plot(weights, avg_acceleration, marker='o', linestyle='-', color='g')
plt.title('Average Acceleration for Weights')
plt.xlabel('Weights')
plt.ylabel('Average Acceleration [m/s$^2$]')
plt.ylim(0.2, 0.5)
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'average_acceleration.png'))
plt.close()
