import socket
import csv
import os
import re

HOST = '0.0.0.0'  # Listen on all available network interfaces
PORT = 3331       # Port number you want your server to listen on
SAVE_DIR = 'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/Dataset'  # Directory where you want to save the file
FILE_NAME = 'p1p3.csv'  # Name of the Excel file

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
                        # Write received data to CSV file
                        file_path = os.path.join(SAVE_DIR, FILE_NAME)
                        with open(file_path, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(numbers)
                        
                        print(numbers)

        print('Connection closed')