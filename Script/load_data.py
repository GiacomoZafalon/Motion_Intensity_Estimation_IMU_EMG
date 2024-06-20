import pandas as pd
import shutil
import os

def copy_files(file_list, directory):
    for file in file_list:
        src = os.path.join(directory, file)
        dest = os.path.join(directory, f"copy_{file}")
        shutil.copy(src, dest)

def process_files(file_list, directory, rows_to_remove):
    for file in file_list:
        file_path = os.path.join(directory, file)
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Remove the specified number of rows
        df = df.iloc[rows_to_remove:]
        
        # Rewrite the first column to start from 0.01 and increment by 0.01
        df.iloc[:, 0] = [round(0.01 * (i + 1), 2) for i in range(len(df))]
        
        # Save the modified DataFrame back to the CSV file
        df.to_csv(file_path, index=False, header=False)

def main():
    tot_persons = 20
    tot_weights = 5
    tot_attempts = 1
    # tot_persons = 1
    # tot_weights = 1
    # tot_attempts = 1


    for person in range(1, tot_persons + 1):
        for weight in range(1, tot_weights + 1):
            for attempt in range(1, tot_attempts + 1):
                # person = 9
                # weight = 5
                # attempt = 1
                # Prompt user for the directory and number of rows to remove
                directory = rf'C:\Users\giaco\OneDrive\Desktop\Università\Tesi_Master\GitHub\Dataset\P{person}\W{weight}\A{attempt}\imu'
                # rows_to_remove = int(input(f"p{person}w{weight}: Enter the number of top rows to remove: "))
                rows_to_remove = 1
                
                file_list = ['sensor1.csv', 'sensor2.csv', 'sensor3.csv', 'sensor4.csv']
                
                # Copy the files
                copy_files(file_list, directory)
                
                # Process the files
                process_files(file_list, directory, rows_to_remove)

if __name__ == "__main__":
    main()
