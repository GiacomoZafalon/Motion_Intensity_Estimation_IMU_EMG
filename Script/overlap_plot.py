# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import os

# # def normalize_data(data, num_points=100):
# #     """
# #     Normalize the length of the data to num_points using interpolation.
# #     """
# #     x = np.linspace(0, 1, len(data))
# #     x_new = np.linspace(0, 1, num_points)
# #     data_interpolated = np.interp(x_new, x, data)
# #     return data_interpolated

# # person = 5
# # # Define the base path for the files
# # base_path = rf'C:\Users\giaco\OneDrive\Desktop\Università\Tesi_Master\GitHub\Dataset\P{person}'

# # # Number of points to interpolate to
# # num_points = 100

# # # Columns to plot
# # columns_to_plot = [1, 3, 4]
# # titles = ['Shoulder Flexion', 'Elbow Flexion', 'Lumbar Flexion']

# # # Iterate over the columns to plot
# # for idx, col in enumerate(columns_to_plot):
# #     # Initialize a figure for each subplot
# #     plt.figure(figsize=(8, 6))
    
# #     # Iterate over the W values from 1, 3, 5
# #     for W in [1, 3, 5]:
# #         # Construct the file path
# #         file_path = os.path.join(base_path, f'W{W}', 'A1', 'imu', 'data_neural.csv')
        
# #         # Load the CSV file
# #         data = pd.read_csv(file_path)
        
# #         # Normalize the data
# #         y_data_normalized = normalize_data(data.iloc[:, col], num_points)
        
# #         # Plot the data
# #         plt.plot(np.linspace(0, 100, num_points), y_data_normalized, label=f'W{W}')
    
# #     # Add labels and legend
# #     plt.xlabel('% of Completion [%]')
# #     plt.ylabel(f'Angle [°]')
# #     plt.title(titles[idx])
# #     plt.legend()
    
# #     # Save the plot
# #     plot_path = rf'C:\Users\giaco\OneDrive\Desktop\Università\Tesi_Master\Results\Overlap_plot\w1w3w5_person{person}_col{col}.png'
# #     plt.savefig(plot_path)
    
# #     # Show the plot
# #     # plt.show()


# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.spatial.transform import Rotation as R

# def euler_to_vector(euler_angles):
#     rotation = R.from_euler('xyz', euler_angles, degrees=True)
#     vector = rotation.apply(np.array([0, 0, 1]))
#     return vector

# def compute_angle_between_vectors(v1, v2):
#     dot_product = np.sum(v1 * v2, axis=1)
#     magnitude_v1 = np.linalg.norm(v1, axis=1)
#     magnitude_v2 = np.linalg.norm(v2, axis=1)
#     cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
#     angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip values to avoid numerical issues
#     return np.degrees(angle)  # Convert radians to degrees

# def plot_average_angles_for_each_weight(base_dir, tot_persons, tot_weights):
#     plt.figure(figsize=(12, 8))

#     for weight in range(1, tot_weights + 1):
#         all_interpolated_angles = []

#         for person in range(1, tot_persons + 1):
#             file_path = os.path.join(base_dir, f'P{person}', f'W{weight}', 'A1', 'imu', 'merged_file_final_filt.csv')
#             if os.path.exists(file_path):
#                 df = pd.read_csv(file_path, header=None)

#                 if df.shape[1] > 42:
#                     euler_angles_v1 = df.iloc[:, [28, 29, 27]].values
#                     euler_angles_v2 = df.iloc[:, [41, 42, 40]].values

#                     vectors_v1 = np.apply_along_axis(euler_to_vector, 1, euler_angles_v1)
#                     vectors_v2 = np.apply_along_axis(euler_to_vector, 1, euler_angles_v2)

#                     angles = compute_angle_between_vectors(vectors_v1, vectors_v2)

#                     normalized_length = np.linspace(0, 100, num=100)
#                     interpolated_angles = np.interp(normalized_length, np.linspace(0, 100, num=len(angles)), angles)
#                     all_interpolated_angles.append(interpolated_angles)
#                 else:
#                     print(f"File {file_path} does not have enough columns.")
#             else:
#                 print(f"File not found: {file_path}")

#         if all_interpolated_angles:
#             average_angles = np.mean(all_interpolated_angles, axis=0)
#             # std_dev_angles = np.std(all_interpolated_angles, axis=0)

#             plt.plot(normalized_length, average_angles, label=f'Weight {weight}', linewidth=2.5)
#             # plt.fill_between(normalized_length, average_angles - std_dev_angles, average_angles + std_dev_angles, alpha=0.3)

#     plt.title('Average Angle Between Vectors for Each Weight')
#     plt.xlabel('Percentage of Movement Completed [%]')
#     plt.ylabel('Angle [°]')
#     plt.ylim(0, 90)
#     plt.grid(True)
#     plt.tight_layout()
#     plt.legend(loc='upper right')
    
#     plot_path = os.path.join(rf'C:\Users\giaco\OneDrive\Desktop\Università\Tesi_Master\Results\Shadow_plots\imu', 'average_angles_for_each_weight_plot.png')
#     # plt.savefig(plot_path)
#     plt.show()
#     plt.close()
#     print(f"Saved plot for average angles for each weight")

# # Example usage
# base_dir = 'C:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/'
# tot_persons = 10  # Replace with actual number of persons
# tot_weights = 5  # Replace with actual number of weights
# plot_average_angles_for_each_weight(base_dir, tot_persons, tot_weights)


# # import os
# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from scipy.spatial.transform import Rotation as R

# # def euler_to_vector(euler_angles):
# #     rotation = R.from_euler('xyz', euler_angles, degrees=True)
# #     # Assuming the direction vector is initially aligned with the x-axis
# #     vector = rotation.apply(np.array([1, 0, 0]))
# #     return vector

# # def compute_angle_between_vectors(v1, v2):
# #     dot_product = np.sum(v1 * v2, axis=1)
# #     magnitude_v1 = np.linalg.norm(v1, axis=1)
# #     magnitude_v2 = np.linalg.norm(v2, axis=1)
# #     cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
# #     angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip values to avoid numerical issues
# #     return np.degrees(angle)  # Convert radians to degrees

# # def plot_all_weights_in_one_plot(base_dir, tot_weights, person):
# #     plt.figure(figsize=(10, 6))

# #     for weight in range(1, tot_weights + 1):
# #         file_path = os.path.join(base_dir, f'P{person}', f'W{weight}', 'A1', 'imu', 'merged_file_final_filt.csv')
# #         if os.path.exists(file_path):
# #             df = pd.read_csv(file_path, header=None)

# #             if df.shape[1] > 42:
# #                 euler_angles_v1 = df.iloc[:, [27, 29, 28]].values
# #                 euler_angles_v2 = df.iloc[:, [40, 42, 41]].values

# #                 vectors_v1 = np.apply_along_axis(euler_to_vector, 1, euler_angles_v1)
# #                 vectors_v2 = np.apply_along_axis(euler_to_vector, 1, euler_angles_v2)

# #                 angles = compute_angle_between_vectors(vectors_v1, vectors_v2)

# #                 normalized_length = np.linspace(0, 100, num=100)
# #                 interpolated_angles = np.interp(normalized_length, np.linspace(0, 100, num=len(angles)), angles)

# #                 plt.plot(normalized_length, interpolated_angles, label=f'Weight {weight}')
# #             else:
# #                 print(f"File {file_path} does not have enough columns.")
# #         else:
# #             print(f"File not found: {file_path}")

# #     plt.title('Angle Between Vectors for Person 1 Across Different Weights')
# #     plt.xlabel('Percentage of Movement Completed [%]')
# #     plt.ylabel('Angle [°]')
# #     plt.ylim(0, 150)
# #     plt.grid(True)
# #     plt.tight_layout()
# #     plt.legend()

# #     plot_path = os.path.join(rf'C:\Users\giaco\OneDrive\Desktop\Università\Tesi_Master\Results\Overlap_plot', f'person1_all_weights_angle_between_vectors_plot.png')
# #     # plt.savefig(plot_path)
# #     plt.show()
# #     plt.close()
# #     print(f"Saved plot for all weights - Angle Between Vectors for Person 1")

# # # Example usage
# # base_dir = 'C:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/'
# # tot_weights = 5  # Replace with actual number of weights
# # person = 21
# # plot_all_weights_in_one_plot(base_dir, tot_weights, person)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_columns_for_all_weights(base_dir, tot_persons, tot_weights, columns):
    # Dictionary to hold the average and std dev data for each body part
    body_part_data = {2: {'name': 'Lower back', 'data': []},
                      15: {'name': 'Torso', 'data': []},
                      29: {'name': 'Upper arm', 'data': []},
                      42: {'name': 'Forearm', 'data': []}}
    
    # Iterate through each column
    for column in columns:
        body_name = ''
        if column in body_part_data:
            body_name = body_part_data[column]['name']
        
        # Iterate through each weight
        for weight in range(1, tot_weights + 1):
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
                    else:
                        print(f"File {file_path} does not have enough columns.")
                else:
                    print(f"File not found: {file_path}")

            # Compute the average and standard deviation of the interpolated data
            if all_interpolated_data:
                average_data = np.mean(all_interpolated_data, axis=0)
                std_dev_data = np.std(all_interpolated_data, axis=0)
                body_part_data[column]['data'].append((weight, average_data, std_dev_data))
    
    # Now, create the plots for each body part
    for column, info in body_part_data.items():
        if info['data']:
            plt.figure(figsize=(10, 6))
            for weight, average_data, std_dev_data in info['data']:
                normalized_length = np.linspace(0, 100, num=100)
                plt.plot(normalized_length, average_data, label=f'Weight {weight}', linewidth=2.5)
                # plt.fill_between(normalized_length, average_data - std_dev_data, average_data + std_dev_data, alpha=0.3)
            
            plt.title(f'{info["name"]} Orientation Across All Weights', fontsize=16)
            plt.xlabel('Percentage of Movement Completed [%]', fontsize=16)
            plt.ylabel('Orientation [°]', fontsize=16)
            plt.grid(True)
            plt.legend(fontsize=14)
            plt.tight_layout()
            
            # Set y-limits based on body part
            if column == 2:
                plt.ylim(0, 95)
            elif column == 15:
                plt.ylim(0, 95)
            elif column == 29:
                plt.ylim(-20, 190)
            elif column == 42:
                plt.ylim(-20, 190)
            
            # Save the plot
            plot_path = os.path.join(rf'C:\Users\giaco\OneDrive\Desktop\Università\Tesi_Master\Results\Shadow_plots\imu', f'bbb_{info["name"]}_orientation_all_weights_plot.png')
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved combined plot for body part: {info['name']}")

# Example usage
base_dir = 'C:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/'
tot_persons = 30
tot_weights = 5
columns = [2, 15, 29, 42]
plot_columns_for_all_weights(base_dir, tot_persons, tot_weights, columns)

