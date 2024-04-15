import socket
import csv
import os
import re
import msvcrt
import numpy as np

HOST = '0.0.0.0'  # Listen on all available network interfaces
PORT = 3331       # Port number you want your server to listen on
SAVE_DIR = 'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset'  # Directory where you want to save the file
FILE_NAME = 'p1p3.csv'  # Name of the CSV file
COLUMN_NAMES = ['time', 'eul_z', 'eul_y', 'eul_x', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z', 'linacc_x', 'linacc_y', 'linacc_z']

# File path
file_path = os.path.join(SAVE_DIR, FILE_NAME)

# Function to write column names to the CSV file
def write_column_names(file_path, column_names):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(column_names)

# Create a TCP/IP socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    # Bind the socket to the address and port
    s.bind((HOST, PORT))
    
    # Listen for incoming connections
    s.listen()

    print(f'Server listening on port {PORT}')
    data_list = []  # List to store received data

    # Write column names to the CSV file
    write_column_names(file_path, COLUMN_NAMES)

    while True:
        # Accept incoming connections
        conn, addr = s.accept()
        with conn:
            print(f'Connected by {addr}')
            
            # Handle incoming data
            while True:
                data = conn.recv(1024)
                # Check for keyboard input
                if msvcrt.kbhit():
                    key = msvcrt.getch().decode()
                    if key == 'm':
                        break  # Exit the loop if 'm' is pressed

                if not data:
                    break
                # Process received data
                lines = data.decode().split('\n')

                # Process each line separately
                for line in lines:
                    # Use regular expression to extract numeric values
                    numbers = re.findall(r'\d+\.\d+', line)
                    
                    # Check if the row contains valid data
                    if len(numbers) == 16:
                        print(numbers)
                        # Append received data to the CSV file
                        with open(file_path, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(numbers)
                        
                        # print(numbers)

        print('Connection closed')
        break

print('Processing data...')
# Read the CSV file and remove the 7th column from each row
columns = {  # Dictionary to store each column's data
    'time': [],
    'eul_z': [],
    'eul_y': [],
    'eul_x': [],
    'acc_x': [],
    'acc_y': [],
    'acc_z': [],
    'gyro_x': [],
    'gyro_y': [],
    'gyro_z': [],
    'mag_x': [],
    'mag_y': [],
    'mag_z': [],
    'linacc_x': [],
    'linacc_y': [],
    'linacc_z': []
}

with open(file_path, mode='r', newline='') as file:
    reader = csv.DictReader(file)
    for row in reader:
        for key, value in row.items():
            columns[key].append(value)

# Assign each column to a variable
time, eul_z, eul_y, eul_x, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z, linacc_x, linacc_y, linacc_z = columns.values()

# Optionally, you can convert the columns to appropriate data types (e.g., float)
time = [float(value) for value in time]
eul_z = [float(value) for value in eul_z]
eul_y = [float(value) for value in eul_y]
eul_x = [float(value) for value in eul_x]
acc_x = [float(value) for value in acc_x]
acc_y = [float(value) for value in acc_y]
acc_z = [float(value) for value in acc_z]
gyro_x = [float(value) for value in gyro_x]
gyro_y = [float(value) for value in gyro_y]
gyro_z = [float(value) for value in gyro_z]
mag_x = [float(value) for value in mag_x]
mag_y = [float(value) for value in mag_y]
mag_z = [float(value) for value in mag_z]
linacc_x = [float(value) for value in linacc_x]
linacc_y = [float(value) for value in linacc_y]
linacc_z = [float(value) for value in linacc_z]

# Define calibration parameters
A = np.array([[1.004332, 0.000046, 0.004896],  # 'A^-1' matrix from Magneto
              [0.000046, 0.969793, 0.009452],
              [0.004896, 0.009452, 1.022384]])
# 'Combined bias (b)' vector from Magneto
b = np.array([0.027031, -0.040204, 0.046558])

# Convert the lists to NumPy arrays
linacc_x = np.array(linacc_x)
linacc_y = np.array(linacc_y)
linacc_z = np.array(linacc_z)

# Stack the arrays column-wise to create the matrix
rawAcc = np.column_stack((linacc_x, linacc_y, linacc_z))
N = len(rawAcc)
calibAcc = np.zeros((N, 3), dtype='float')
for i in range(N):
    currMeas = np.array([rawAcc[i, 0], rawAcc[i, 1], rawAcc[i, 2]])
    calibAcc[i, :] = A @ (currMeas - b)

# Create lists for calibrated acceleration components
linacc_calib_x = calibAcc[:, 0].tolist()
linacc_calib_y = calibAcc[:, 1].tolist()
linacc_calib_z = calibAcc[:, 2].tolist()

print('Done')