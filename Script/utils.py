import os
import shutil
import pandas as pd
import re
from scipy.signal import butter, filtfilt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def extract_columns_from_csv(tot_persons, tot_weights, tot_attempts):
    column_to_euler_acc_gyro = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18, 19, 20, 21, 22, 27, 28, 29, 30, 31, 32, 33, 34, 35, 40, 41, 42, 43, 44, 45, 46, 47, 48]
    columns_to_extract = [column_to_euler_acc_gyro]
    output_file = ['data_neural_euler_acc_gyro.csv']

    for person in range(1, tot_persons + 1):
        for weight in range(1, tot_weights + 1):
            for attempt in range(1, tot_attempts + 1):
                file_path = f'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/P{person}/W{weight}/A{attempt}/imu/merged_file_final_2_filt.csv'
                
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
            plt.title(f'Weight {weight} - {body} orientation', fontsize=16)
            plt.xlabel('Percentage of Movement Completed [%]', fontsize=16)
            plt.ylabel(f'Orientation [°]', fontsize=16)
            plt.grid(True)
            plt.tight_layout()
            plt.legend(fontsize=14)  # Only includes labeled plots

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
                        trunc_length = np.int64(len(column_data)*0.6)
                        # Normalize the length (truncate at 60% and rescale to 100%)
                        normalized_length = np.linspace(0, 100, num=60)
                        interpolated_data = np.interp(normalized_length, np.linspace(0, 100, num=trunc_length), column_data[:trunc_length])
                        # interpolated_data = interpolated_data[:60]
                        # print(np.shape(normalized_length), np.shape(interpolated_data))
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
            plt.title(f'Weight {weight} - EMG {body}', fontsize=16)
            plt.xlabel('Percentage of Movement Completed [%]', fontsize=16)
            plt.ylabel(f'EMG Value [mV]', fontsize=16)
            plt.xlim(0, 100)  # Adjust x-axis to show 0% to 100%
            plt.ylim(-50, 700)
            plt.grid(True)
            plt.tight_layout()
            plt.legend(fontsize=14)  # Only includes labeled plots

            # Save the plot
            plot_path = os.path.join(r'C:\Users\giaco\OneDrive\Desktop\Università\Tesi_Master\Results\Shadow_plots\emg', f'weight_{weight}_emg_{body}_plot.png')
            plt.savefig(plot_path)
            plt.close()
            # plt.show()
            # aaa
            print(f"Saved plot for weight {weight} and EMG {body}")

def plot_confusion_matrix_with_std_dev(data, std_dev, labels, predictions, precisions, plot_path=None):
    accuracies = [data[i][i] for i in range(len(data))]
    # Create extended matrix with accuracy column and precision row
    extended_matrix = np.full((data.shape[0] + 2, data.shape[1] + 2), np.nan)

    # Fill extended matrix with mean values
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            extended_matrix[i, j] = data[i, j]

    # Fill accuracy column
    for i in range(data.shape[0]):
        extended_matrix[i, -1] = accuracies[i]

    # Fill precision row
    for j in range(data.shape[1]):
        extended_matrix[-1, j] = precisions[j]

    # Create a figure and a set of subplots
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot the heatmap
    heatmap = sns.heatmap(extended_matrix, annot=True, fmt=".1f", cmap="Blues", cbar=False, ax=ax, annot_kws={"size": 14})

    # Format annotations to include percentage symbol
    for t in ax.texts:
        t.set_text(t.get_text() + "%")

    # Get the color map from the heatmap
    cmap = heatmap.collections[0].cmap
    norm = mcolors.Normalize(vmin=np.nanmin(extended_matrix), vmax=np.nanmax(extended_matrix))

    # Function to calculate brightness/luminance of a color
    def calculate_luminance(color):
        # Convert RGB to grayscale luminance
        luminance = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]
        return luminance

    # Add standard deviation annotations with contrasting text color
    for i in range(extended_matrix.shape[0]):
        for j in range(extended_matrix.shape[1]):
            if not np.isnan(std_dev[i, j]):
                text = f"\n\n+/- {std_dev[i, j]:.1f}"
                cell_color = cmap(norm(extended_matrix[i, j]))  # Get the color of the cell
                cell_color_rgb = mcolors.to_rgba(cell_color)[:3]  # Convert to RGB tuple

                # Calculate luminance/brightness of the cell color
                luminance = calculate_luminance(cell_color_rgb)

                # Choose text color based on luminance
                if luminance > 0.6:  # Bright background, use black text
                    text_color = '#000000'
                else:  # Dark background, use white text
                    text_color = '#ffffff'

                ax.text(j + 0.5, i + 0.5, text, ha='center', va='center', color=text_color, fontsize=12)

    # Remove the ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Add label row on top
    for col in range(len(predictions)):
        ax.text(col + 0.5, -0.25, predictions[col], ha='center', va='center', color='black', fontsize=16)

    # Add label column on the left
    for row in range(len(labels)):
        ax.text(-0.5, row + 0.5, labels[row], ha='center', va='center', color='black', fontsize=16)

    # Add accuracy column label on the top
    ax.text(len(predictions) + 1.5, -0.25, 'Accuracy', ha='center', va='center', color='black', fontsize=16)

    # Add precision row label on the left
    ax.text(-0.5, len(labels) + 1.5, 'Precision', ha='center', va='center', color='black', fontsize=16)

    # Save or show the plot
    if plot_path:
        plt.savefig(plot_path)
        print(plot_path)
    else:
        plt.show()



tot_persons = 30
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

# # Gets the average duration, average acceleration, and maximum acceleration throughout all the folders
results = compute_average_lengths_and_accelerations(base_dir, tot_persons, tot_weights)
for result in results:
    print(f"Weight {result['weight']} -> Avg duration: {result['avg_length']:.2f}s; Avg Acceleration: {result['avg_acc']:.2f}; Max Acceleration: {result['max_acc']:.2f} at P{result['person_max_acc']}; Avg Max Acceleration: {result['avg_max_acc']:.2f}")

# # Plots all the Euler data for each weight in one plot with average and std dev
# columns_imu = [2, 15, 29, 42]  # Columns to be plotted
# plot_columns_for_all_weights(base_dir, tot_persons, tot_weights, columns_imu)

# # Plots all the emg data for each weight in one plot with average and std dev
# columns_emg = [0, 1, 2]  # EMG columns to be plotted
# plot_emg_columns_for_all_weights(base_dir, tot_persons, tot_weights, columns_emg)

# # Plots the confusion matrix for dataset A, B, and C
# # DATASET A
# data = np.array([[96.9, 3.1, 0, 0, 0],[7.7, 66.5, 16.4, 9.4, 0],[1.3, 9.2, 83.5, 6, 0],[1.5, 5.3, 11.7, 75.2, 6.3],[2.3, 2.4, 1.3, 3.7, 90.3]])
# std_dev = np.array([[4.4, 4.4, 0, 0, 0, np.nan, 4.4],[7.8, 5.9, 6.5, 6.6, 0, np.nan, 5.9],[1.8, 6.4, 4.8, 5.4, 0, np.nan, 4.8],[2.1, 2.7, 0.1, 8.1, 9, np.nan, 8.1],[2.4, 2.7, 1.8, 2.7, 4.5, np.nan, 4.5],[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],[6.4, 5.5, 5.0, 1.8, 7.0, np.nan, np.nan]])
# precisions = [86.8, 75.7, 74.2, 79.8, 95.0]
# path = r'C:\Users\giaco\OneDrive\Desktop\Università\Tesi_Master\Results\Confusion_matrices\conf_mat_A.png'
# # DATASET B
# data = np.array([[97.0, 0.6, 1.3, 1.0, 0.1],[21.2, 25.8, 22.5, 26.6, 3.9],[0, 4.4, 75.9, 13.4, 6.2],[1.9, 1.9, 12.7, 74.2, 9.3],[9.9, 2.7, 2.1, 0.1, 85.1]])
# std_dev = np.array([[2.8, 0.6, 2.5, 1.9, 0.2, np.nan, 2.8],[4.7, 3.5, 5.4, 5.0, 2.7, np.nan, 3.5],[0, 3.8, 6.8, 4.5, 4.4, np.nan, 6.8],[0.6, 1.8, 1.2, 3.4, 2.3, np.nan, 3.4],[4.6, 3.5, 2.3, 0.2, 2.9, np.nan, 2.9],[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],[3.6, 10.2, 2.3, 2.8, 5.9, np.nan, np.nan]])
# precisions = [67.6, 71.8, 66.3, 67.4, 84.5]
# path = r'C:\Users\giaco\OneDrive\Desktop\Università\Tesi_Master\Results\Confusion_matrices\conf_mat_B.png'
# # DATASET C
# data = np.array([[97.3, 0.0, 2.7, 0.0, 0.0],[14.4, 40.2, 27.9, 14.7, 3.4],[0.0, 15.7, 74.4, 7.6, 2.2],[1.8, 8.0, 15.8, 66.3, 8.1],[8.7, 0.1, 0.0, 1.0, 90.3]])
# std_dev = np.array([[3.8, 0.0, 3.8, 0.0, 0.0, np.nan, 3.8],[8.5, 5.4, 8.4, 2.0, 3.1, np.nan, 5.4],[0.0, 6.6, 8.9, 3.9, 1.3, np.nan, 8.9],[1.8, 1.2, 4.2, 3.7, 1.8, np.nan, 3.7],[4.3, 0.1, 0.0, 1.7, 3.3, np.nan, 3.3],[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],[5.8, 9.9, 6.2, 4.1, 4.5, np.nan, np.nan]])
# precisions = [72.6, 63.1, 61.6, 75.5, 89.0]
# path = r'C:\Users\giaco\OneDrive\Desktop\Università\Tesi_Master\Results\Confusion_matrices\conf_mat_C.png'
# labels = ['Label 0', 'Label 1', 'Label 2', 'Label 3', 'Label 4']
# predictions = ['Pred. 0', 'Pred. 1', 'Pred. 2', 'Pred. 3', 'Pred. 4']
# plot_confusion_matrix_with_std_dev(data, std_dev, labels, predictions, precisions, plot_path=path)
