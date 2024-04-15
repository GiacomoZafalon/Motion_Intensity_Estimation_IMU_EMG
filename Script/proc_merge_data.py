import csv
import os

# Define the paths of the input CSV files for each sensor
sensor_files = {
    'sensor1': 'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/sensor1.csv',
    'sensor2': 'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/sensor2.csv',
    'sensor3': 'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/sensor3.csv',
    'sensor4': 'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/sensor4.csv'
}

# Define the path of the output merged CSV file
merged_file = 'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/merged_data.csv'

def merge_csv_files(sensor_files, merged_file):
    # Open the output CSV file for writing
    with open(merged_file, mode='w', newline='') as outfile:
        writer = csv.writer(outfile)

        # Open the input CSV file for sensor 1
        with open(sensor_files['sensor1'], mode='r') as file1:
            reader1 = csv.reader(file1)

            # Iterate through the rows of sensor 1
            for row1 in reader1:
                first_value = row1[0]

                # Look for matching rows in the other CSV files
                matching_row2 = find_matching_row(first_value, sensor_files['sensor2'])
                matching_row3 = find_matching_row(first_value, sensor_files['sensor3'])
                matching_row4 = find_matching_row(first_value, sensor_files['sensor4'])

                # If matching rows are found in all files, merge and write to the output CSV file
                if matching_row2 and matching_row3 and matching_row4:
                    merged_row = row1 + matching_row2 + matching_row3 + matching_row4
                    writer.writerow(merged_row)

def find_matching_row(value, filename):
    # Open the CSV file for reading
    with open(filename, mode='r') as file:
        reader = csv.reader(file)

        # Iterate through the rows of the CSV file
        for row in reader:
            # Check if the first value of the row matches the target value
            if row[0] == value:
                return row

    # If no matching row is found, return None
    return None

# Call the function to merge the CSV files
merge_csv_files(sensor_files, merged_file)

print('Merged data saved to', merged_file)
