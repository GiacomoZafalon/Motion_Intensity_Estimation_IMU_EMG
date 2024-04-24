import socket
import csv
import os
import re
import msvcrt
import numpy as np
from save_dir_info import person, weight, attempt

HOST = '0.0.0.0'  # Listen on all available network interfaces
PORT = 3335       # Port number you want your server to listen on
FILE_NAME = 'emg_data.csv'  # Name of the CSV file

person = 1
weight = 1
attempt = 1

if FILE_NAME == 'emg_mvc.csv':
    SAVE_DIR = f'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/P{person}'
    file_path = os.path.join(SAVE_DIR, FILE_NAME)
else:
    SAVE_DIR = f'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/P{person}/W{weight}/A{attempt}/emg'
    file_path = os.path.join(SAVE_DIR, FILE_NAME)

# Create the save directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

# Create a TCP/IP socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    # Bind the socket to the address and port
    s.bind((HOST, PORT))
    
    # Listen for incoming connections
    s.listen()

    print(f'Server listening on port {PORT}')
    data_list = []  # List to store received data

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

                    # Append received data to the CSV file
                    with open(file_path, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(numbers)
                        print(numbers)

        print('Connection closed')
        break
