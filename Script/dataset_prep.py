import pickle
import torch
import pandas as pd
import os

class DataProcessor:
    def __init__(self, person, weight, attempt, length, time):
        self.person = person
        self.weight = weight
        self.attempt = attempt
        self.time = time
        self.length = length

    def load_imu_data(self, person, weight, attempt, length, time):
        if time == 0:
            imu_path = rf'C:\Users\giaco\OneDrive\Desktop\Università\Tesi_Master\Dataset_train_augmented\data_neural_euler_acc_gyro_P{person}_W{weight}_A{attempt}.csv'
        if time == 1:
            imu_path = rf'C:\Users\giaco\OneDrive\Desktop\Università\Tesi_Master\Dataset_test_augmented\data_neural_euler_acc_gyro_P{person}_W{weight}_A{attempt}.csv'
        imu_data = pd.read_csv(imu_path, header=None)  # Load CSV data using pandas

        imu_chunks = []
        num_timesteps = imu_data.shape[0]
        num_chunks = num_timesteps // length
        remainder = num_timesteps % length

        for i in range(num_chunks, 0, -1):
            chunk = imu_data.iloc[(i - 1) * length: i * length]
            imu_chunks.append(torch.tensor(chunk.values))

        if remainder > 15 and num_chunks > 0:
            last_chunk = imu_data.iloc[:length]
            imu_chunks.append(torch.tensor(last_chunk.values))

        if num_chunks == 0 and remainder > 0:
            # Pad the remainder with the first row's values
            last_chunk = imu_data.iloc[:remainder]
            pad_length = length - remainder
            pad_values = imu_data.iloc[0].values
            pad_array = pd.DataFrame([pad_values] * pad_length)
            last_chunk = pd.concat([pad_array, imu_data.iloc[:remainder]], ignore_index=True)
            last_chunk.iloc[:, 0] = 0.01 + 0.01 * last_chunk.index
            imu_chunks.append(torch.tensor(last_chunk.values))

        return imu_chunks

    def load_emg_data(self, person, weight, attempt, time):
        if time == 0:
            emg_path = rf'C:\Users\giaco\OneDrive\Desktop\Università\Tesi_Master\Dataset_train_augmented\emg_label_P{person}_W{weight}_A{attempt}.csv'
        if time == 1:
            emg_path = rf'C:\Users\giaco\OneDrive\Desktop\Università\Tesi_Master\Dataset_test_augmented\emg_label_P{person}_W{weight}_A{attempt}.csv'
        emg_data = pd.read_csv(emg_path)  # Load CSV data using pandas
        emg_tensor = torch.tensor(emg_data.iloc[:, 1].values)  # Convert DataFrame to tensor
        return emg_tensor

    def process_data(self, output_file, save_interval=10):
        all_data = []

        for p in range(1, self.person + 1):
            for w in range(1, self.weight + 1):
                for a in range(1, self.attempt + 1):
                    emg_data = self.load_emg_data(p, w, a, self.time)
                    imu_data = self.load_imu_data(p, w, a, self.length, self.time)
                    for chunk in imu_data:
                        all_data.append((chunk, emg_data))

            # Save to pickle file every `save_interval` people
            if p % save_interval == 0:
                self.append_data(output_file, all_data)
                all_data = []  # Clear the list after saving

            # Save current person number to text file
            self.save_progress(output_file, p)

            print(f'Person {p}/{self.person} done')

        # Save any remaining data
        if all_data:
            self.append_data(output_file, all_data)

    def append_data(self, output_file, data):
        mode = 'ab' if os.path.exists(output_file) else 'wb'
        with open(output_file, mode) as f:
            for item in data:
                pickle.dump(item, f)

    def load_data(self, input_file):
        data = []
        with open(input_file, 'rb') as f:
            while True:
                try:
                    data.append(pickle.load(f))
                except EOFError:
                    break
        return data

    def save_progress(self, output_file, person_number):
        progress_file = os.path.join(os.path.dirname(output_file), 'progress.txt')
        with open(progress_file, 'w') as f:
            f.write(f'Last processed person: {person_number}')


for i in range(2):
    if i == 0:
        # Set parameters
        person = 3311
        weight = 5
        attempt = 10
        length = 50
        output_file = r'C:\Users\giaco\OneDrive\Desktop\Università\Tesi_Master\GitHub\All_data_file\all_data_train_trunc.pkl'

        # Process data and save to file
        processor = DataProcessor(person, weight, attempt, length, i)
        processor.process_data(output_file, save_interval=10)  # Save data every 10 people
    if i == 1:
        # Set parameters
        person = 1510
        weight = 5
        attempt = 10
        length = 50
        output_file = r'C:\Users\giaco\OneDrive\Desktop\Università\Tesi_Master\GitHub\All_data_file\all_data_test_trunc.pkl'

        # Process data and save to file
        processor = DataProcessor(person, weight, attempt, length, i)
        processor.process_data(output_file, save_interval=10)  # Save data every 10 people
