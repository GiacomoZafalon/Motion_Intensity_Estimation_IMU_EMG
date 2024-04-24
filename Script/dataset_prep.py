import torch
import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self, person, weight, attempt):
        self.person = person
        self.weight = weight
        self.attempt = attempt

    def load_imu_data(self, person, weight, attempt):
        imu_path = f'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/P{person}/W{weight}/A{attempt}/imu/data_neural.csv'
        imu_data = pd.read_csv(imu_path)  # Load CSV data using pandas

        imu_chunks = []
        num_timesteps = imu_data.shape[0]
        num_chunks = num_timesteps // 100
        remainder = num_timesteps % 100

        for i in range(num_chunks):
            chunk = imu_data.iloc[i * 100: (i + 1) * 100]
            imu_chunks.append(torch.tensor(chunk.values))

        if remainder > 20:
            last_chunk = imu_data.iloc[-100:]
            imu_chunks.append(torch.tensor(last_chunk.values))

        return imu_chunks

    def load_emg_data(self, person, weight, attempt):
        emg_path = f'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/P{person}/W{weight}/A{attempt}/emg/emg_label.csv'
        emg_data = pd.read_csv(emg_path)  # Load CSV data using pandas
        emg_tensor = torch.tensor(emg_data.iloc[:, 1])  # Convert DataFrame to tensor
        return emg_tensor

    def process_data(self):
        all_data = []

        for p in range(1, self.person + 1):
            for w in range(1, self.weight + 1):
                for a in range(1, self.attempt + 1):
                    emg_data = self.load_emg_data(p, w, a)
                    imu_data = self.load_imu_data(p, w, a)
                    for chunk in range(len(imu_data)):
                        imu_chunk = imu_data[chunk]
                        all_data.append((imu_chunk, emg_data))

            print(f'Person {p}/{self.person} done')

        return all_data

# Example usage:
person = 1
weight = 1
attempt = 1

processor = DataProcessor(person, weight, attempt)
all_data = processor.process_data()
print(len(all_data))
for i in range(len(all_data)):
    data, label = all_data[i]
    print(data.shape, label.shape)
# print(data)
# print(label)