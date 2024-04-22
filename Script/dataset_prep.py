import torch
import pandas as pd

class DataProcessor:
    def __init__(self, person, weight, attempt):
        self.person = person
        self.weight = weight
        self.attempt = attempt

    def load_imu_data(self, person, weight, attempt):
        imu_path = f'c:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/GitHub/Dataset/P{person}/W{weight}/A{attempt}/imu/data_neural.csv'
        imu_data = pd.read_csv(imu_path)  # Load CSV data using pandas
        imu_tensor = torch.tensor(imu_data.values)  # Convert DataFrame to tensor
        return imu_tensor

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
                    imu_data = self.load_imu_data(p, w, a)
                    emg_data = self.load_emg_data(p, w, a)

                    all_data.append((imu_data, emg_data))

            print(f'Person {p}/{self.person} done')

        return all_data

# Example usage:
person = 1
weight = 1
attempt = 1

processor = DataProcessor(person, weight, attempt)
all_data = processor.process_data()
# data, label = all_data[0]
# print(label)