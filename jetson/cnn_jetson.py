from idna import alabel
import time
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import ConcatDataset as ConcatDataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import datetime as dt
import sklearn
from sklearn.preprocessing import MinMaxScaler
if torch.cuda.is_available():
    print(torch.cuda.device_count())
    print('Training runs on GPU')
    device = torch.device("cuda:1")
else:
    print('no GPU available')
    device = torch.device("cpu")
# device = torch.device('cpu')

#hyper parameters
train_hours = 180 * 24 #day*24h
valid_hours = 30 * 24 #day*24h
test_hours = 30 * 24 #day*24h
timesteps = 1 * 24 #day*24h
batch_size = 64

consumption_value_list = [random.choice(range(1, 10)) for _ in range(180*24)]

#normalization
consumption_value_list = sklearn.preprocessing.minmax_scale(consumption_value_list)
consumption_value_list = consumption_value_list.tolist()
# print(hourly_consumption_list)


def train_test(consumption_value_list, train_hours, test_hours):
    train_list = consumption_value_list[:train_hours]
    test_list = consumption_value_list[train_hours:train_hours+test_hours]
    return train_list, test_list

def timestep_split(source_list):
    num = len(source_list) - timesteps
    input_sequences = []
    labels = []
    for i in range(num):
        input_sequence = []
        for j in range(timesteps):
            input_sequence.append(source_list[i+j])
        input_sequences.append(input_sequence)
        labels.append(source_list[i+timesteps])
    input_sequences = np.array(input_sequences)
    labels = np.array(labels)
    return input_sequences, labels


class Dataset(nn.Module):
    def __init__(self, input_sequences, labels):
        self.input_sequences = input_sequences
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        input_sequence = self.input_sequences[index]
        label = self.labels[index]
        input_sequence = np.float32(input_sequence)
        label = np.float32(label)
        input_sequence = torch.from_numpy(input_sequence).requires_grad_(True)
        input_sequence = input_sequence.reshape(1, len(input_sequence))
        return input_sequence, label


train_list, test_valid_list = train_test(consumption_value_list, train_hours, test_hours + valid_hours)
train_inputs, train_labels = timestep_split(train_list)
train_set = Dataset(train_inputs, train_labels)
train_loader = DataLoader(train_set, batch_size = batch_size, drop_last = True)

#Hyper parameters
conv1_out = 50
conv2_out = 50
kernel1 = 16
kernel2 = 8
dropout = 0.2
class Net(nn.Module):
    def __init__(self, timesteps):
        super(Net, self).__init__()
        self.dropout = nn.Dropout(p = dropout)
        self.conv1 = nn.Conv1d(in_channels = 1, out_channels = conv1_out, kernel_size = kernel1)
        self.conv2 = nn.Conv1d(in_channels = conv1_out, out_channels = conv2_out, kernel_size = kernel2)
        self.outlen1 = (timesteps - kernel1) + 1
        self.outlen2 = (self.outlen1 - kernel2) + 1
        self.fc = nn.Linear(self.outlen2 * conv2_out, 1)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = x.reshape(len(x), -1)
        output = self.fc(x).squeeze(-1)
        return output

my_model = Net(timesteps).to(device)
my_model.train()
criterion = nn.MSELoss()
optimizer = optim.Adam(my_model.parameters(), lr = 0.001)
epochs = 300
patience = 0
min_patience = 300

train_losses = []
average_train_losses = []
average_valid_losses = []
start_time = time.time()
for epoch in range(epochs):
  #=============train==================
    my_model.train()
    for sequence, label in train_loader:
        sequence = sequence.to(device)
        label = label.to(device)
        output = my_model(sequence)
        optimizer.zero_grad()
        loss = criterion(output, label).to(torch.float32)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    train_loss = np.average(train_losses) 
    average_train_losses.append(train_loss)
    print('epoch', epoch, 'train_loss', train_loss)
    train_losses = []

total_training_time = time.time() - start_time
time_per_epoch = total_training_time / (epoch + 1)
print('=====Training results=====')
print('Total time for training: ', total_training_time, '\n', 'Time per epoch', time_per_epoch)
with open('cnn.txt', 'a') as f:
    print('Total time for training: ', total_training_time, '\n', 'Time per epoch', time_per_epoch, file = f)
