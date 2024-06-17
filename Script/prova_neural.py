import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pickle
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

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class CNN_LSTM(nn.Module):
    def __init__(self, num_data, out_channels=64, kernel_conv=3, kernel_pool=2, hidden_lstm=64, layers_lstm=2, num_classes=5):
        super(CNN_LSTM, self).__init__()
        self.batch_size = batch_size

        # 1D Convolutional Layer
        self.conv1 = nn.Conv1d(in_channels=num_data, out_channels=out_channels, kernel_size=kernel_conv)
        self.pool1 = nn.MaxPool1d(kernel_size=kernel_pool)

        # LSTM layers
        self.lstm = nn.LSTM(out_channels, hidden_lstm, layers_lstm, batch_first=True)
        
        # Fully connected layer for regression
        self.fc1 = nn.Linear(hidden_lstm*hidden_lstm, num_classes)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 1e-1)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # print('1', np.shape(x))
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, num_data, time_steps)
        # print('2', np.shape(x))
        x = F.relu(self.conv1(x))
        # print('3', np.shape(x))
        x = self.pool1(x)
        # print('4', np.shape(x))
        x, _ = self.lstm(x)
        # print('5', np.shape(x))
        x = x.flatten()
        x = x.view(batch_size, -1)
        # print('6', np.shape(x))
        x = self.fc1(x)  # Use only the last output of LSTM
        # print('7', np.shape(x))
        return F.softmax(x, dim=1)

def train(model, train_loader, criterion, optimizer, epochs=3):
    model.train()
    losses = []
    total_batches = math.ceil(len(train_loader.dataset) / batch_size)
    for epoch in range(epochs):
        running_loss = 0.0
        print('-----------EPOCH %d-----------' % (epoch + 1))
        for i, (inputs, labels) in enumerate(train_loader):
            if len(train_loader.dataset) % batch_size != 0 and i == total_batches - 1:
                break
            optimizer.zero_grad()
            outputs = model(inputs.float())
            labels = labels.flatten()
            # print(labels)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i + 1) % 400 == 0:
                print('[episode: %d] loss: %f' % (i + 1, running_loss / 400))
                losses.append(running_loss / 400)
                running_loss = 0.0

    plt.plot(losses)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.show()

def test(model, test_loader, criterion):
    model.eval()
    total_error = 0
    total_batches = math.ceil(len(test_loader.dataset) / batch_size)
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            if len(test_loader.dataset) % batch_size != 0 and i == total_batches - 1:
                break
            outputs = model(inputs.float())
            labels = labels.flatten()
            loss = criterion(outputs, labels.long())
            total_error += loss.item()
            if (i + 1) % 400 == 0:
                print(outputs, labels)
    average_loss = total_error / len(test_loader)
    print('Average test loss: {:.4f}'.format(average_loss))

if __name__ == '__main__':
    person = 10021
    weight = 5
    attempt = 6

    # Load `all_data` from the file
    input_file = r'C:\Users\giaco\OneDrive\Desktop\Università\Tesi_Master\GitHub\All_data_file\all_data.pkl'
    processor = DataProcessor(person, weight, attempt)
    # Load data (for testing or further processing)
    print('Loading data...')
    all_data = processor.load_data(input_file)
    # print(f'Total data loaded: {len(loaded_data)}')

    print(len(all_data))

    data_list = [data for data, label in all_data]
    label_list = [label for data, label in all_data]

    num_rows = len(data_list)
    indices = torch.randperm(num_rows).tolist()
    shuffled_data = [data_list[i] for i in indices]
    shuffled_labels = [label_list[i] for i in indices]

    shuffled_dataset = CustomDataset(list(zip(shuffled_data, shuffled_labels)))

    batch_size = 64
    train_set = DataLoader(shuffled_dataset[:len(shuffled_dataset) // 2], batch_size=batch_size, shuffle=True, num_workers=2)
    test_set = DataLoader(shuffled_dataset[len(shuffled_dataset) // 2:], batch_size=batch_size, shuffle=False, num_workers=2)

    num_data = np.shape(shuffled_data[0])[1]
    time_steps = np.shape(shuffled_data[0])[0]
    model = CNN_LSTM(num_data, layers_lstm=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train(model, train_set, criterion, optimizer, epochs=3)

    print('Training done. Now testing...')

    test(model, test_set, criterion)
