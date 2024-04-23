import pandas as pd
import numpy as np

def interpolate_and_adjust_time(file_path, num_interpolations=1):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path, header=None)

    contracted_time = []

    # Initialize a variable to track the time offset
    inter = 0

    # Iterate through the original DataFrame
    for i in range(len(df) - 1):
        if (i + 1) % 10 != 0:
            append_value = df.iloc[i].values.tolist() - np.array([0.01*inter] + [0]*19)
            contracted_time.append(append_value)
        else:
            inter += 1

    # Add the last row of the original DataFrame
    contracted_time.append(df.iloc[-1].values.tolist() - np.array([0.01*inter] + [0]*19))

    # Create DataFrame from the interpolated data
    interpolated_df = pd.DataFrame(contracted_time, columns=df.columns)

    return interpolated_df

# Example usage:
person = 1
weight = 1
attempt = 1
file_path = f'C:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/P{person}/W{weight}/A{attempt}/imu/sensor1.csv'
interpolated_df = interpolate_and_adjust_time(file_path)

# Save the updated DataFrame to a new CSV file if needed
output_file_path = f'C:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/P{person}/W{weight}/A{attempt}/imu/sensor1_contracted_time.csv'
interpolated_df.to_csv(output_file_path, header=False, index=False)
