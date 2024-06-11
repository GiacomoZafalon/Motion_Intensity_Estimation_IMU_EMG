import pandas as pd

tot_person = 5
tot_weight = 5
tot_attempt = 6

for person in range(1, tot_person + 1):
    for weight in range(1, tot_weight + 1):
        for attempt in range(1, tot_attempt + 1):

            # Define the path to your CSV file
            file_path = f'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/P{person}/W{weight}/A{attempt}/imu/merged_data.csv'

            # Define the columns to be extracted (0-based index)
            columns_to_euler = [0, 1, 2, 3, 15, 16, 17, 29, 30, 31, 43, 44, 45]
            columns_to_euler_acc = [0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 21, 22, 23, 29, 30, 31, 35, 36, 37, 43, 44, 45, 49, 50, 51]
            columns_to_euler_gyro = [0, 1, 2, 3, 4, 5, 6, 15, 16, 17, 18, 19, 20, 29, 30, 31, 32, 33, 34, 43, 44, 45, 46, 47, 48]
            column_to_euler_acc_gyro = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20, 21, 22, 23, 29, 30, 31, 32, 33, 34, 35, 36, 37, 43, 44, 45, 46, 47, 48, 49, 50, 51]

            columns_to_extract = [columns_to_euler, columns_to_euler_acc, columns_to_euler_gyro, column_to_euler_acc_gyro]
            output_file = ['data_neural_euler.csv', 'data_neural_euler_acc.csv', 'data_neural_euler_gyro.csv', 'data_neural_euler_acc_gyro.csv']
            # Read the CSV file
            for i in range(len(columns_to_extract)):
                data = pd.read_csv(file_path, usecols=columns_to_extract[i])

                # Define the output file path
                output_file_path = f'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/P{person}/W{weight}/A{attempt}/imu/{output_file[i]}'

                # Save the extracted columns to a new CSV file
                data.to_csv(output_file_path, index=False, header=False)

            print(f'Data saved for person {person}/{tot_person}, weight {weight}/{tot_weight} for attempt {attempt}/{tot_attempt}')

print('Done')