import os
import shutil
import pandas as pd
import re
from scipy.signal import butter, filtfilt

def extract_columns_from_csv(tot_persons, tot_weights, tot_attempts):
    column_to_euler_acc_gyro = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18, 19, 20, 21, 22, 27, 28, 29, 30, 31, 32, 33, 34, 35, 40, 41, 42, 43, 44, 45, 46, 47, 48]
    columns_to_extract = [column_to_euler_acc_gyro]
    output_file = ['data_neural_euler_acc_gyro.csv']

    for person in range(1, tot_persons + 1):
        for weight in range(1, tot_weights + 1):
            for attempt in range(1, tot_attempts + 1):
                file_path = f'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/P{person}/W{weight}/A{attempt}/imu/merged_file_final_filt.csv'
                
                for i in range(len(columns_to_extract)):
                    try:
                        data = pd.read_csv(file_path, usecols=columns_to_extract[i], header=None)
                    except FileNotFoundError:
                        print(f"File not found: {file_path}")
                        continue
                    except pd.errors.EmptyDataError:
                        print(f"No data: {file_path}")
                        continue
                    
                    output_file_path = f'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset_original_2/P{person}/W{weight}/A{attempt}/imu/{output_file[i]}'
                    data.to_csv(output_file_path, index=False, header=None)

    print('Data saved')





def find_highest_person_number(folder_path):
    # Regular expression pattern to extract the person number from the filename
    pattern = re.compile(r'data_neural_euler_acc_gyro_P(\d+)_W\d+_A\d+\.csv')
    
    highest_person = 0
    
    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            person_number = int(match.group(1))
            if person_number > highest_person:
                highest_person = person_number

    print(f"The highest person number is: {highest_person}")

def copy_and_rename_files(base_dir, output_dir, number_to_add):
    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    for person_dir in os.listdir(base_dir):
        person_path = os.path.join(base_dir, person_dir)
        if not os.path.isdir(person_path):
            continue
        
        # Assuming person_dir is in the format 'P1', 'P2', etc., we extract the number and add 8
        person_number = int(person_dir[1:]) + number_to_add
        new_person_dir = f'P{person_number}'
        
        for weight_dir in os.listdir(person_path):
            weight_path = os.path.join(person_path, weight_dir)
            if not os.path.isdir(weight_path):
                continue
            
            for attempt_dir in os.listdir(weight_path):
                attempt_path = os.path.join(weight_path, attempt_dir)
                if not os.path.isdir(attempt_path):
                    continue
                
                imu_source = os.path.join(attempt_path, 'imu', 'data_neural_euler_acc_gyro.csv')
                emg_source = os.path.join(attempt_path, 'emg', 'emg_label.csv')

                if os.path.exists(imu_source):
                    imu_dest = os.path.join(output_dir, f'data_neural_euler_acc_gyro_{new_person_dir}_{weight_dir}_{attempt_dir}.csv')
                    shutil.copy(imu_source, imu_dest)
                    # print(f"Copied {imu_source} to {imu_dest}")

                if os.path.exists(emg_source):
                    emg_dest = os.path.join(output_dir, f'emg_label_{new_person_dir}_{weight_dir}_{attempt_dir}.csv')
                    shutil.copy(emg_source, emg_dest)
                    print(f"Copied {emg_source} to {emg_dest}")


def delete_extra_attempt_folders(base_directory, tot_persons, tot_weights):
    for person in range(1, tot_persons + 1):
        for weight in range(1, tot_weights + 1):
            directory = os.path.join(base_directory, f'P{person}', f'W{weight}')
            for attempt_folder in os.listdir(directory):
                attempt_path = os.path.join(directory, attempt_folder)
                if os.path.isdir(attempt_path) and attempt_folder.startswith('A'):
                    attempt_number = int(attempt_folder[1:])
                    if attempt_number > 1:
                        shutil.rmtree(attempt_path)
                        print(f"Deleted folder: {attempt_path}")

def get_average_lengths(base_dir, tot_persons, tot_weights):
    # Initialize a dictionary to hold lengths for each weight
    lengths = {f'W{w}': [] for w in range(1, tot_weights + 1)}

    # Iterate through each person and weight
    for person in range(1, tot_persons + 1):
        for weight in range(1, tot_weights + 1):
            file_path = os.path.join(base_dir, f'P{person}', f'W{weight}', 'A1', 'imu', 'merged_file_final_filt.csv')
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, header=None)
                lengths[f'W{weight}'].append(len(df))

    # Calculate the average lengths for each weight
    average_lengths = {weight: (sum(lengths[weight]) / len(lengths[weight])) if lengths[weight] else 0 for weight in lengths}

    return average_lengths

def compute_max_accelerations(data_dir, tot_persons, tot_weights):
    weight_results = {weight: {'total_max': 0, 'count': 0} for weight in range(1, tot_weights + 1)}

    for person in range(1, tot_persons + 1):
        for weight in range(1, tot_weights + 1):
            path = os.path.join(data_dir, f'P{person}', f'W{weight}', 'A1', 'imu')
            merged_file_path = os.path.join(path, 'merged_file_final_filt.csv')

            if os.path.exists(merged_file_path):
                df_final = pd.read_csv(merged_file_path, header=None)
                accel_columns = [4, 5, 6, 17, 18, 19, 30, 31, 32, 43, 44, 45]

                # Compute average and maximum accelerations for the current file
                max_accel = df_final[accel_columns].max().max()

                weight_results[weight]['total_max'] += max_accel
                weight_results[weight]['count'] += 1
            else:
                print(f"File not found: {merged_file_path}")

    # Calculate the final average and maximum accelerations for each weight
    results = []
    for weight in range(1, tot_weights + 1):
        if weight_results[weight]['count'] > 0:
            max_accel = weight_results[weight]['total_max'] / weight_results[weight]['count']
            results.append({
                'weight': weight,
                'maximum_acceleration': max_accel
            })
        else:
            results.append({
                'weight': weight,
                'maximum_acceleration': None
            })

    return results

tot_persons = 28
tot_weights = 5
tot_attempts = 10
number_to_add = 0

base_dir = 'C:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/'
output_dir = 'C:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/Dataset_train_augmented/'

## Uncomment the needed function ##

# Extracts columns from merged_file to create data_neural_euler_acc_gyro.csv
# extract_columns_from_csv(tot_persons, tot_weights, tot_attempts)

# Deletes the non useful attempt folders
# delete_extra_attempt_folders(base_dir, tot_persons, tot_weights)

# Copies the files data_neural_euler_acc_gyro.csv from the p/w/a folders to the combined folder with the new name
# copy_and_rename_files(base_dir, output_dir, number_to_add)

# Finds the highest person number in the augmented dataset folder
# find_highest_person_number(output_dir)

# Get the average lengths
average_lengths = get_average_lengths(base_dir, tot_persons, tot_weights)
for weight, avg_length in average_lengths.items():
    print(f'Weight: {weight}, Average length: {avg_length:.2f}')

results = compute_max_accelerations(base_dir, tot_persons, tot_weights)
for result in results:
    print(f"Weight: W{result['weight']}, Maximum Acceleration: {result['maximum_acceleration']:.2f}")

