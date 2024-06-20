# import os
# import shutil

# # Source and destination folders
# data_dir = 'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset'

# # Total counts for persons, weights, and attempts
# tot_person = 20
# tot_weight = 5
# tot_attempt = 1

# # Iterate through all person, weight, and attempt combinations
# for person in range(11, tot_person + 1):
#     for weight in range(1, tot_weight + 1):
#         for attempt in range(1, tot_attempt + 1):
            
#             # Define paths to IMU and EMG folders
#             emg_original = f'{data_dir}/P1/W{weight}/A1/emg/emg_label.csv'
#             emg_dest_dir = f'{data_dir}/P{person}/W{weight}/A{attempt}/emg'
#             emg_dest = os.path.join(emg_dest_dir, 'emg_label.csv')
            
#             # Create destination directory if it does not exist
#             if not os.path.exists(emg_dest_dir):
#                 os.makedirs(emg_dest_dir)
            
#             # Copy the file
#             if not os.path.exists(emg_dest):
#                 shutil.copy(emg_original, emg_dest)

#     print(f'person {person}/{tot_person} done')

# print('Done')

# import pandas as pd

# tot_persons = 20
# tot_weights = 5
# tot_attempts = 1
# # tot_persons = 1
# # tot_weights = 1
# # tot_attempts = 1


# for person in range(1, tot_persons + 1):
#     for weight in range(1, tot_weights + 1):
#         for attempt in range(1, tot_attempts + 1):

#             # person = 1
#             # weight = 1
#             # attempt = 1

#             # Define the path to your CSV file
#             file_path = f'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/P{person}/W{weight}/A{attempt}/imu/merged_file_final.csv'

#             # Define the columns to be extracted (0-based index)
#             # columns_to_euler = [0, 1, 2, 3, 15, 16, 17, 29, 30, 31, 43, 44, 45]
#             # columns_to_euler_acc = [0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 21, 22, 23, 29, 30, 31, 35, 36, 37, 43, 44, 45, 49, 50, 51]
#             # columns_to_euler_gyro = [0, 1, 2, 3, 4, 5, 6, 15, 16, 17, 18, 19, 20, 29, 30, 31, 32, 33, 34, 43, 44, 45, 46, 47, 48]
#             column_to_euler_acc_gyro = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18, 19, 20, 21, 22, 27, 28, 29, 30, 31, 32, 33, 34, 35, 40, 41, 42, 43, 44, 45, 46, 47, 48]

#             columns_to_extract = [column_to_euler_acc_gyro]
#             output_file = ['data_neural_euler_acc_gyro.csv']
#             # Read the CSV file
#             for i in range(len(columns_to_extract)):
#                 data = pd.read_csv(file_path, usecols=columns_to_extract[i], header=None)

#                 # Define the output file path
#                 output_file_path = f'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/P{person}/W{weight}/A{attempt}/imu/{output_file[i]}'

#                 # Save the extracted columns to a new CSV file
#                 data.to_csv(output_file_path, index=False, header=None)

# print('Data saved')

import os
import shutil

base_dir = 'C:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/'
output_dir = 'C:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/Dataset_aug/'

def copy_and_rename_files(base_dir, output_dir):
    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    for person_dir in os.listdir(base_dir):
        person_path = os.path.join(base_dir, person_dir)
        if not os.path.isdir(person_path):
            continue
        
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
                    imu_dest = os.path.join(output_dir, f'data_neural_euler_acc_gyro_{person_dir}_{weight_dir}_{attempt_dir}.csv')
                    shutil.copy(imu_source, imu_dest)
                    print(f"Copied {imu_source} to {imu_dest}")

                if os.path.exists(emg_source):
                    emg_dest = os.path.join(output_dir, f'emg_label_{person_dir}_{weight_dir}_{attempt_dir}.csv')
                    shutil.copy(emg_source, emg_dest)
                    print(f"Copied {emg_source} to {emg_dest}")

# Execute the function
copy_and_rename_files(base_dir, output_dir)

