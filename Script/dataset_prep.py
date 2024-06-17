import pickle
import torch
import pandas as pd
import os

class DataProcessor:
    def __init__(self, person, weight, attempt):
        self.person = person
        self.weight = weight
        self.attempt = attempt

    def load_imu_data(self, person, weight, attempt):
        imu_path = rf'C:\Users\giaco\OneDrive\Desktop\Università\Tesi_Master\GitHub\Dataset\data_neural_p{person}_w{weight}_a{attempt}.csv'
        imu_data = pd.read_csv(imu_path)  # Load CSV data using pandas

        imu_chunks = []
        num_timesteps = imu_data.shape[0]
        num_chunks = num_timesteps // 200
        remainder = num_timesteps % 200

        for i in range(num_chunks, 0, -1):
            chunk = imu_data.iloc[(i - 1) * 200: i * 200]
            imu_chunks.append(torch.tensor(chunk.values))

        if remainder > 50:
            last_chunk = imu_data.iloc[:200]
            imu_chunks.append(torch.tensor(last_chunk.values))

        return imu_chunks

    def load_emg_data(self, person, weight, attempt):
        emg_path = rf'C:\Users\giaco\OneDrive\Desktop\Università\Tesi_Master\GitHub\Dataset\emg_label_p{person}_w{weight}_a{attempt}.csv'
        emg_data = pd.read_csv(emg_path)  # Load CSV data using pandas
        emg_tensor = torch.tensor(emg_data.iloc[:, 1].values)  # Convert DataFrame to tensor
        return emg_tensor

    def process_data(self, output_file, save_interval=10):
        all_data = []

        for p in range(1, self.person + 1):
            for w in range(1, self.weight + 1):
                for a in range(1, self.attempt + 1):
                    emg_data = self.load_emg_data(p, w, a)
                    imu_data = self.load_imu_data(p, w, a)
                    for chunk in imu_data:
                        all_data.append((chunk, emg_data))

            # Save to pickle file every `save_interval` people
            if p % save_interval == 0:
                self.append_data(output_file, all_data)
                all_data = []  # Clear the list after saving

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

# Set parameters
person = 10021
weight = 5
attempt = 6
output_file = r'C:\Users\giaco\OneDrive\Desktop\Università\Tesi_Master\GitHub\All_data_file\all_data.pkl'

# Process data and save to file
processor = DataProcessor(person, weight, attempt)
processor.process_data(output_file, save_interval=10)  # Save data every 10 people

# Load data (for testing or further processing)
# loaded_data = processor.load_data(output_file)
# print(f'Total data loaded: {len(loaded_data)}')
