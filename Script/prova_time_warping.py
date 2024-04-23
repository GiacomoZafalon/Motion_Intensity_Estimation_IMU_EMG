import pandas as pd
import numpy as np

def time_warping(file_path, type=2, amount_of_warping=10):
    if type == 1:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path, header=None)

        # Define the interpolation function
        def interpolate_row(prev_row, next_row, num_interpolations=1):
            interpolated_rows = []
            for i in range(num_interpolations):
                alpha = (i + 1) / (num_interpolations + 1)
                interpolated_row = [prev_row[0]]  # Keep the first value (time) unchanged
                for j in range(1, len(prev_row)):
                    interpolated_value = prev_row[j] + (next_row[j] - prev_row[j]) * alpha
                    interpolated_row.append(interpolated_value)
                # Normalize the quaternion values
                quaternion_values = np.array(interpolated_row[-4:])
                norm = np.linalg.norm(quaternion_values)
                normalized_quaternion = quaternion_values / norm
                interpolated_row[-4:] = normalized_quaternion
                # interpolated_row.extend(normalized_quaternion)
                interpolated_rows.append(interpolated_row)
            return interpolated_rows

        # Initialize an empty DataFrame to store the interpolated data
        interpolated_data = []

        # Initialize a variable to track the time offset
        inter = 0

        # Iterate through the original DataFrame
        for i in range(len(df) - 1):
            append_value = df.iloc[i].values.tolist() + np.array([round(0.01*inter, 2)] + [0]*19)
            interpolated_data.append(append_value)
            if (i + 1) % amount_of_warping == 0:
                inter += 1
                interpolated_rows = interpolate_row(df.iloc[i].values.tolist(), df.iloc[i+1].values.tolist())
                for row in interpolated_rows:
                    interpolated_data.append(row + np.array([round(0.01*inter, 2)] + [0]*19))

        # Add the last row of the original DataFrame
        interpolated_data.append(df.iloc[-1].values.tolist() + np.array([round(0.01*inter, 2)] + [0]*19))

        # Create DataFrame from the interpolated data
        warped_df = pd.DataFrame(interpolated_data, columns=df.columns)
        # Round the timestamp column to 2 decimal places
        warped_df[0] = warped_df[0].round(2)


    elif type == 2:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path, header=None)

        contracted_time = []

        # Initialize a variable to track the time offset
        inter = 0

        # Iterate through the original DataFrame
        for i in range(len(df) - 1):
            if (i + 1) % amount_of_warping != 0:
                append_value = df.iloc[i].values.tolist() - np.array([round(0.01*inter, 2)] + [0]*19)
                contracted_time.append(append_value)
            else:
                inter += 1

        # Add the last row of the original DataFrame
        contracted_time.append(df.iloc[-1].values.tolist() - np.array([round(0.01*inter, 2)] + [0]*19))

        # Create DataFrame from the interpolated data
        warped_df = pd.DataFrame(contracted_time, columns=df.columns)
        # Round the timestamp column to 2 decimal places
        warped_df[0] = warped_df[0].round(2)

    return warped_df

# Example usage:
person = 1
weight = 1
attempt = 1
file_path = f'C:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/P{person}/W{weight}/A{attempt}/imu/sensor1.csv'
interpolated_df = time_warping(file_path, 1, 5)

# Save the updated DataFrame to a new CSV file if needed
output_file_path = f'C:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/P{person}/W{weight}/A{attempt}/imu/sensor1_interpolated_time_adjusted.csv'
interpolated_df.to_csv(output_file_path, header=False, index=False)
