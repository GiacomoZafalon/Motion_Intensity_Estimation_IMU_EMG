import os
import shutil
import pandas as pd
import re
from scipy.signal import butter, filtfilt
import numpy as np
import matplotlib.pyplot as plt

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
                    
                    output_file_path = f'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/P{person}/W{weight}/A{attempt}/imu/{output_file[i]}'
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

def compute_average_lengths_and_accelerations(base_dir, tot_persons, tot_weights):
    # Initialize dictionaries to hold lengths and acceleration results for each weight
    lengths = {f'W{w}': [] for w in range(1, tot_weights + 1)}
    weight_results = {
        weight: {
            'total_average': 0,
            'count': 0,
            'max_accel_person': None,
            'max_accel_value': -float('inf'),
            'max_accels': []
        } for weight in range(1, tot_weights + 1)
    }

    # Iterate through each person and weight
    for person in range(1, tot_persons + 1):
        for weight in range(1, tot_weights + 1):
            file_path = os.path.join(base_dir, f'P{person}', f'W{weight}', 'A1', 'imu', 'merged_file_final_filt.csv')
            if os.path.exists(file_path):
                # Calculate length
                df = pd.read_csv(file_path, header=None)
                lengths[f'W{weight}'].append(len(df))

                # Calculate accelerations
                path = os.path.join(base_dir, f'P{person}', f'W{weight}', 'A1', 'imu')
                merged_file_path = os.path.join(path, 'merged_file_final_filt.csv')

                if os.path.exists(merged_file_path):
                    df_final = pd.read_csv(merged_file_path, header=None)
                    accel_columns = [4, 5, 6, 17, 18, 19, 30, 31, 32, 43, 44, 45]

                    # Compute average and maximum accelerations for the current file
                    average_accel = df_final[accel_columns].mean().mean()
                    max_accel = df_final[accel_columns].max().max()

                    # Update the results for the current weight
                    weight_results[weight]['total_average'] += average_accel
                    weight_results[weight]['count'] += 1

                    # Store the maximum acceleration for averaging later
                    weight_results[weight]['max_accels'].append(max_accel)

                    # Check if this person has the max acceleration for this weight
                    if max_accel > weight_results[weight]['max_accel_value']:
                        weight_results[weight]['max_accel_value'] = max_accel
                        weight_results[weight]['max_accel_person'] = person
                else:
                    print(f"File not found: {merged_file_path}")
            else:
                print(f"File not found: {file_path}")

    # Calculate the average lengths for each weight
    average_lengths = {weight: (sum(lengths[weight]) / len(lengths[weight])) if lengths[weight] else 0 for weight in lengths}

    # Calculate the final average and maximum accelerations for each weight
    results = []
    for weight in range(1, tot_weights + 1):
        if weight_results[weight]['count'] > 0:
            average_accel = weight_results[weight]['total_average'] / weight_results[weight]['count']
            max_accel = weight_results[weight]['max_accel_value']
            avg_of_max_accels = sum(weight_results[weight]['max_accels']) / len(weight_results[weight]['max_accels'])
            max_accel_person = weight_results[weight]['max_accel_person']
            results.append({
                'weight': f'W{weight}',
                'avg_length': average_lengths[f'W{weight}']/100,
                'avg_acc': average_accel,
                'max_acc': max_accel,
                'avg_max_acc': avg_of_max_accels,
                'person_max_acc': max_accel_person
            })
        else:
            results.append({
                'weight': f'W{weight}',
                'avg_length': average_lengths[f'W{weight}']/100,
                'avg_acc': None,
                'max_acc': None,
                'avg_max_acc': None,
                'person_max_acc': None
            })

    return results

def plot_columns_for_all_weights(base_dir, tot_persons, tot_weights, columns):
    # Iterate through each column
    for column in columns:
        # Iterate through each weight
        for weight in range(1, tot_weights + 1):
            plt.figure(figsize=(10, 6))

            all_interpolated_data = []

            # Iterate through each person
            for person in range(1, tot_persons + 1):
                file_path = os.path.join(base_dir, f'P{person}', f'W{weight}', 'A1', 'imu', 'merged_file_final_filt.csv')
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path, header=None)

                    # Check if the dataframe has enough columns
                    if df.shape[1] > column:
                        column_data = df.iloc[:, column]
                        # Normalize the length
                        normalized_length = np.linspace(0, 100, num=100)
                        interpolated_data = np.interp(normalized_length, np.linspace(0, 100, num=len(column_data)), column_data)
                        all_interpolated_data.append(interpolated_data)
                        plt.plot(normalized_length, interpolated_data, alpha=0.2)  # No label here
                    else:
                        print(f"File {file_path} does not have enough columns.")
                else:
                    print(f"File not found: {file_path}")

            # Compute the average and standard deviation of the interpolated data
            if all_interpolated_data:
                average_data = np.mean(all_interpolated_data, axis=0)
                std_dev_data = np.std(all_interpolated_data, axis=0)

                plt.plot(normalized_length, average_data, label='Average', color='black', linewidth=2.5)
                plt.plot(normalized_length, average_data + std_dev_data, 'r--', linewidth=2, label='+/- Std Dev')
                plt.plot(normalized_length, average_data - std_dev_data, 'r--', linewidth=2)

                # Fill the area between the +1 and -1 standard deviation lines
                plt.fill_between(normalized_length, average_data - std_dev_data, average_data + std_dev_data, color='gray', alpha=0.3)

            if column == 2:
                body = 'Lower back'
                plt.ylim(0, 95)
            elif column == 15:
                body = 'Torso'
                plt.ylim(0, 95)
            elif column == 29:
                body = 'Upper arm'
                plt.ylim(-20, 190)
            elif column == 42:
                body = 'Forearm'
                plt.ylim(-20, 190)
            plt.title(f'Weight {weight} - {body} orientation')
            plt.xlabel('Percentage of Movement Completed [%]')
            plt.ylabel(f'{body} orientation [°]')
            plt.grid(True)
            plt.tight_layout()
            plt.legend()  # Only includes labeled plots

            # Save the plot
            plot_path = os.path.join(rf'C:\Users\giaco\OneDrive\Desktop\Università\Tesi_Master\Results\Shadow_plots\imu', f'weight_{weight}_column_{column}_{body}_plot.png')
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved plot for weight {weight} and body: {body}")

def plot_emg_columns_for_all_weights(base_dir, tot_persons, tot_weights, columns):
    # Iterate through each column
    for column in columns:
        # Iterate through each weight
        for weight in range(1, tot_weights + 1):
            plt.figure(figsize=(10, 6))

            all_interpolated_data = []

            # Iterate through each person
            for person in range(1, tot_persons + 1):
                file_path = os.path.join(base_dir, f'P{person}', f'W{weight}', 'A1', 'emg', 'emg_data.csv')
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path, header=None)

                    # Check if the dataframe has enough columns
                    if df.shape[1] > column:
                        column_data = df.iloc[:, column]
                        # Normalize the length
                        normalized_length = np.linspace(0, 100, num=100)
                        interpolated_data = np.interp(normalized_length, np.linspace(0, 100, num=len(column_data)), column_data)
                        all_interpolated_data.append(interpolated_data)
                        plt.plot(normalized_length, interpolated_data, alpha=0.2)  # No label here
                    else:
                        print(f"File {file_path} does not have enough columns.")
                else:
                    print(f"File not found: {file_path}")

            # Compute the average and standard deviation of the interpolated data
            if all_interpolated_data:
                average_data = np.mean(all_interpolated_data, axis=0)
                std_dev_data = np.std(all_interpolated_data, axis=0)

                plt.plot(normalized_length, average_data, label='Average', color='black', linewidth=2.5)
                plt.plot(normalized_length, average_data + std_dev_data, 'r--', linewidth=2, label='+/- Std Dev')
                plt.plot(normalized_length, average_data - std_dev_data, 'r--', linewidth=2)

                # Fill the area between the +1 and -1 standard deviation lines
                plt.fill_between(normalized_length, average_data - std_dev_data, average_data + std_dev_data, color='gray', alpha=0.3)

            if column == 0:
                body = 'Bicep'
            elif column == 1:
                body = 'Middle deltoid'
            elif column == 2:
                body = 'Front deltoid'
            plt.title(f'Weight {weight} - EMG {body}')
            plt.xlabel('Percentage of Movement Completed [%]')
            plt.ylabel(f'EMG {body} Value [mV]')
            plt.ylim(-50, 700)
            plt.grid(True)
            plt.tight_layout()
            plt.legend()  # Only includes labeled plots

            # Save the plot
            plot_path = os.path.join(rf'C:\Users\giaco\OneDrive\Desktop\Università\Tesi_Master\Results\Shadow_plots\emg', f'weight_{weight}_emg_{body}_plot.png')
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved plot for weight {weight} and EMG {body}")

tot_persons = 10
tot_weights = 5
tot_attempts = 1
number_to_add = 0

base_dir = 'C:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/'
output_dir = 'C:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/Dataset_test_augmented/'

## UNCOMMENT THE NEEDED FUNCTION ##

# # Extracts columns from merged_file to create data_neural_euler_acc_gyro.csv
# extract_columns_from_csv(tot_persons, tot_weights, tot_attempts)

# # Deletes the non useful attempt folders
# delete_extra_attempt_folders(base_dir, tot_persons, tot_weights)

# # Copies the files data_neural_euler_acc_gyro.csv from the p/w/a folders to the combined folder with the new name
# copy_and_rename_files(base_dir, output_dir, number_to_add)

# # Finds the highest person number in the augmented dataset folder
# find_highest_person_number(output_dir)

# Gets the maximum acceleration for each weight
# results = compute_average_lengths_and_accelerations(base_dir, tot_persons, tot_weights)
# for result in results:
#     print(f"Weight {result['weight']} -> Avg duration: {result['avg_length']:.2f}s; Avg Acceleration: {result['avg_acc']:.2f}; Max Acceleration: {result['max_acc']:.2f} at P{result['person_max_acc']}; Avg Max Acceleration: {result['avg_max_acc']:.2f}")

# # Plots all the Euler data for each weight in one plot with average and std dev
# columns_imu = [2, 15, 29, 42]  # Columns to be plotted
# plot_columns_for_all_weights(base_dir, tot_persons, tot_weights, columns_imu)

# # Plots all the emg data for each weight in one plot with average and std dev
columns_emg = [0, 1, 2]  # EMG columns to be plotted
plot_emg_columns_for_all_weights(base_dir, tot_persons, tot_weights, columns_emg)