import socket
import csv
import os
import re
import msvcrt
import numpy as np

HOST = '0.0.0.0'  # Listen on all available network interfaces
PORT = 3332       # Port number you want your server to listen on
SAVE_DIR = 'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset'  # Directory where you want to save the file
FILE_NAME = 'sensor2.csv'  # Name of the CSV file
# COLUMN_NAMES = ['time', 'eul_z', 'eul_y', 'eul_x', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z', 'linacc_x', 'linacc_y', 'linacc_z']

# File path
file_path = os.path.join(SAVE_DIR, FILE_NAME)

# # Function to write column names to the CSV file
# def write_column_names(file_path, column_names):
#     with open(file_path, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(column_names)

# Create a TCP/IP socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    # Bind the socket to the address and port
    s.bind((HOST, PORT))
    
    # Listen for incoming connections
    s.listen()

    print(f'Server listening on port {PORT}')
    data_list = []  # List to store received data

    # Write column names to the CSV file
    # write_column_names(file_path, COLUMN_NAMES)

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
