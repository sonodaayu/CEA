from idna import alabel
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import ConcatDataset as ConcatDataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchmetrics import MeanAbsolutePercentageError as MAPE
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib as mpl
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

df = pd.read_csv('../DATA/edf_data_hard.csv')

#hyper parameters
train_hours = 180 * 24 #day*24h
valid_hours = 30 * 24 #day*24h
test_hours = 30 * 24 #day*24h
timesteps = 1 * 24 #day*24h
batch_size = 64

def accumulated_consumption(df):
    hours = ['0h', '1h', '2h', '3h', '4h', '5h', '6h', '7h', '8h', '9h', '10h', '11h', '12h', '13h', '14h', '15h', '16h', '17h', '18h', '19h', '20h', '21h', '22h', '23h']
    last_value_list = []
    for idx, df_day in df.groupby(['day_count']):
        for hour in hours:
            for idx, df_hour in df_day.groupby(hour):
                if df_hour.iloc[0][hour] == 1:
                    last_value_list.append(df_hour.iloc[-1]['V40'])
                    break
                else:
                    continue
    return last_value_list

#normalization
accumulated_consumption_list = accumulated_consumption(df)
accumulated_consumption_list = sklearn.preprocessing.minmax_scale(accumulated_consumption_list)
accumulated_consumption_list = accumulated_consumption_list.tolist()
# print(hourly_consumption_list)


def train_test(accumulated_consumption_list, train_hours, test_hours):
    train_list = accumulated_consumption_list[:train_hours]
    test_list = accumulated_consumption_list[train_hours:train_hours+test_hours]
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


train_list, test_valid_list = train_test(accumulated_consumption_list, train_hours, test_hours + valid_hours)
test_list, valid_list = train_test(test_valid_list, test_hours, valid_hours)
train_inputs, train_labels = timestep_split(train_list)
valid_inputs, valid_labels = timestep_split(valid_list)
test_inputs, test_labels = timestep_split(test_list)
train_set = Dataset(train_inputs, train_labels)
valid_set = Dataset(valid_inputs, valid_labels)
test_set = Dataset(test_inputs, test_labels)
train_loader = DataLoader(train_set, batch_size = batch_size, drop_last = True)
valid_loader = DataLoader(valid_set, batch_size = len(valid_set))
test_loader = DataLoader(test_set, batch_size = len(test_set))
test_loader_graph = DataLoader(test_set, batch_size = 1)

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
  #=============valid==================
    my_model.eval()
    for sequence, label in valid_loader:
        sequence = sequence.to(device)
        label = label.to(device)
        output = my_model(sequence)
        loss = criterion(output, label).to(torch.float32)
        valid_loss = loss.item()
    train_loss = np.average(train_losses)
    if epoch == 0:
        best_valid_loss = valid_loss
        running_patience = 0
        best_epoch = 0
        torch.save(my_model.to('cpu').state_dict(),  'best-model-parameters-cnn-accumulate.pt')
        my_model.to(device)
    else:
        if valid_loss < best_valid_loss:
            running_patience = 0
            best_valid_loss = valid_loss
            best_epoch = epoch
            torch.save(my_model.to('cpu').state_dict(),  'best-model-parameters-cnn-accumulate.pt')
            my_model.to(device)
        else:
            running_patience += 1
    if epoch > min_patience:
        if running_patience > patience:
            break
    average_train_losses.append(train_loss)
    average_valid_losses.append(valid_loss)
    print('epoch', epoch, 'train_loss', train_loss, 'valid_loss', valid_loss)
    train_losses = []

total_training_time = time.time() - start_time
time_per_epoch = total_training_time / (epoch + 1)
print('=====Training results=====')
print('Best valid loss: ', best_valid_loss, '\n', 'Best epoch: ', best_epoch, '\n', 'Total time for training: ', total_training_time, '\n', 'Time per epoch', time_per_epoch)
# print('Best valid loss: ', best_valid_loss, '\n', 'Best epoch: ', best_epoch, '\n', 'Total time for training: ', total_training_time, '\n', 'Time per epoch', time_per_epoch)
# print('Best epoch: ', best_epoch)
# print('Total time for training: ', total_training_time)
# print('Time per epoch', time_per_epoch)
with open('cnn_accum_results.txt', 'a') as f:
    print('===============================================================', file = f)
    print('=====Training results=====', '\n', 'Best valid loss: ', best_valid_loss, '\n', 'Best epoch: ', best_epoch, '\n', 'Total time for training: ', total_training_time, '\n', 'Time per epoch', time_per_epoch, file = f)

criterion_MAE = nn.L1Loss()
criterion_MSE = nn.MSELoss()
criterion_MAPE = MAPE()
my_model.load_state_dict(torch.load('./best-model-parameters-cnn-accumulate.pt'))
my_model.eval()
my_model.to('cpu')
for sequence, label in test_loader:
    output = my_model(sequence)
    loss_MAE = criterion_MAE(output, label)
    loss_MSE = criterion_MSE(output, label)
    loss_RMSE = torch.sqrt(criterion_MSE(output, label))
    output = output.reshape(len(output))
    loss_MAPE = criterion_MAPE(output, label)

    print('=====Test results=====')
    print('MAE loss:',loss_MAE.item())
    print('MSE loss:',loss_MSE.item())
    print('RMSE loss:',loss_RMSE)
    print('MAPE: ', loss_MAPE)
with open('cnn_accum_results.txt', 'a') as f:
    print('=====Test results=====', '\n', 'MAE loss: ', loss_MAE.item(), '\n', 'MSE loss:',loss_MSE.item(), '\n', 'RMSE loss:',loss_RMSE, '\n', 'Time per epoch', time_per_epoch, 'MAPE: ', loss_MAPE, file = f)

#Inference time on GPU
my_model = my_model.to(device)
start_time = time.time()
for sequence, label in test_loader_graph:
    sequence = sequence.to(device)
    output = my_model(sequence)
total_inferrence_time = time.time() - start_time
inferrence_time_per_one = total_inferrence_time / len(test_loader_graph)
print('Time for inferrence: ', inferrence_time_per_one)
with open('cnn_accum_results.txt', 'a') as f:
    print('=====Inferrence=====', file = f)
    print('Time for inferrence: ', inferrence_time_per_one, file = f)


# #Inference time on CPU
# start_time = time.time()
# for sequence, label in test_loader_graph:
#     output = my_model(sequence)
# total_inferrence_time = time.time() - start_time
# inferrence_time_per_one = total_inferrence_time / len(test_loader_graph)
# print('Time for inferrence: ', inferrence_time_per_one)
# with open('cnn_accum_results.txt', 'a') as f:
#     print('=====Inferrence=====', file = f)
#     print('Time for inferrence: ', inferrence_time_per_one, file = f)