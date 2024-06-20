import os
import shutil

# Source and destination folders
data_dir = 'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset'

# Total counts for persons, weights, and attempts
tot_person = 10
tot_weight = 5
tot_attempt = 1

# Iterate through all person, weight, and attempt combinations
for person in range(1, tot_person + 1):
    for weight in range(1, tot_weight + 1):
        for attempt in range(1, tot_attempt + 1):
            
            # Define paths to IMU and EMG folders
            emg_original = f'{data_dir}/P1/W{weight}/A1/emg/emg_label.csv'
            emg_dest_dir = f'{data_dir}/P{person}/W{weight}/A{attempt}/emg'
            emg_dest = os.path.join(emg_dest_dir, 'emg_label.csv')
            
            # Create destination directory if it does not exist
            if not os.path.exists(emg_dest_dir):
                os.makedirs(emg_dest_dir)
            
            # Copy the file
            if not os.path.exists(emg_dest):
                shutil.copy(emg_original, emg_dest)

    print(f'person {person}/{tot_person} done')

print('Done')
