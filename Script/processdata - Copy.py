import os
import csv
import shutil

# Directory where CSV files are located
data_dir = 'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset'

# List of CSV files
csv_files = ['sensor1.csv', 'sensor2.csv', 'sensor3.csv', 'sensor4.csv']

# Directory to store copies of CSV files
backup_dir = 'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Backup'

# Create the backup directory if it doesn't exist
os.makedirs(backup_dir, exist_ok=True)

# Copy original CSV files to the backup directory
for csv_file in csv_files:
    original_file_path = os.path.join(data_dir, csv_file)
    backup_file_path = os.path.join(backup_dir, csv_file)
    shutil.copyfile(original_file_path, backup_file_path)

# List to store last values from the first column of each CSV file
last_values = []

# Read the last value from the first column of each CSV file
for csv_file in csv_files:
    file_path = os.path.join(data_dir, csv_file)
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        last_row = None
        for row in reader:
            last_row = row  # Update last row iteratively
        last_value = float(last_row[0]) if last_row else None  # Extract last value from the first column
        last_values.append(last_value)  # Append last value to the list

last_values = [int(x * 100) for x in last_values]

# # Find the highest value among the last values
min_value = min(last_values)

# # Calculate the difference between the highest value and each last value
differences = [value - min_value for value in last_values]

# Indices of columns to be removed (0-indexed)
columns_to_remove = [4, 5, 6, 10, 11, 12]

# Calculate the difference between the first value of the row and the first value of the first column in each file
for csv_file in csv_files:
    file_path = os.path.join(data_dir, csv_file)
    with open(file_path, mode='r') as file:
        lines = file.readlines()
    first_row = lines[0].strip().split(',')
    first_value = float(first_row[0])
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write rows back to the file, updating the first column with the calculated differences rounded to two decimal places
        for j, line in enumerate(lines):
            values = line.strip().split(',')
            row_first_value = float(values[0])
            difference = row_first_value - first_value + 0.01
            rounded_difference = round(difference, 2)  # Round the difference to two decimal places
            values[0] = str(rounded_difference)  # Update the first value
            # Remove specified columns
            updated_values = [value for index, value in enumerate(values) if index not in columns_to_remove]
            writer.writerow(updated_values)
