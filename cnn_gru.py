from idna import alabel
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
import matplotlib.pyplot as plt
import matplotlib as mpl
import sklearn
from sklearn.preprocessing import MinMaxScaler
if torch.cuda.is_available():
    print(torch.cuda.device_count())
    print('Training runs on GPU')
    device = torch.device("cuda:0")
else:
    print('no GPU available')
    device = torch.device("cpu")

df = pd.read_csv('../DATA/edf_data_hard.csv')

#hyper parameters
train_hours = 180 * 24 #day*24h
valid_hours = 30 * 24 #day*24h
test_hours = 30 * 24 #day*24h
timesteps = 7 * 24 #day*24h
batch_size = 3

def hourly_consumption(df):
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
    hourly_consumption_list = []
    current_consumption = 0
    for i in range(len(last_value_list)):
        if i == 0:
            hourly_consumption_list.append(last_value_list[0])
            current_consumption = last_value_list[0]
        elif current_consumption > last_value_list[i]:
            current_consumption = last_value_list[i]
            hourly_consumption_list.append(last_value_list[i])
        else:
            current_consumption = last_value_list[i]
            hourly_consumption_list.append(last_value_list[i] - last_value_list[i-1])
    return hourly_consumption_list


#normalization
hourly_consumption_list = hourly_consumption(df)
hourly_consumption_list = sklearn.preprocessing.minmax_scale(hourly_consumption_list)
hourly_consumption_list = hourly_consumption_list.tolist()
# print(hourly_consumption_list)


def train_test(hourly_consumption_list, train_hours, test_hours):
    train_list = hourly_consumption_list[:train_hours]
    test_list = hourly_consumption_list[train_hours:train_hours+test_hours]
    return train_list, test_list

def timestep_split(source_list):
    num = len(source_list) - timesteps
    inputs = []
    labels = []
    for i in range(num):
        single_input = []
        for j in range(timesteps):
            single_input.append(source_list[i+j])
        inputs.append(single_input)
        labels.append(source_list[i+timesteps])
    inputs = np.array(inputs)
    labels = np.array(labels)
    return inputs, labels


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


train_list, test_valid_list = train_test(hourly_consumption_list, train_hours, test_hours + valid_hours)
test_list, valid_list = train_test(test_valid_list, test_hours, valid_hours)
train_inputs, train_labels = timestep_split(train_list)
valid_inputs, valid_labels = timestep_split(valid_list)
test_inputs, test_labels = timestep_split(test_list)
train_set = Dataset(train_inputs, train_labels)
valid_set = Dataset(valid_inputs, valid_labels)
test_set = Dataset(test_inputs, test_labels)
train_loader = DataLoader(train_set, batch_size = batch_size, drop_last = True)
valid_loader = DataLoader(valid_set)
test_loader = DataLoader(test_set, batch_size = len(test_set))

for sequence, label in train_loader:
    print('sequence.shape', sequence.shape)
    print('len(label)', len(label))
    break

#Hyper parameters
conv1_out = 1
conv2_out = 1
kernel1 = 16
kernel2 = 8
hidden_size = 10
num_layers = 2
dropout = 0.2
class Net(nn.Module):
    def __init__(self, timesteps):
        super(Net, self).__init__()
        self.dropout = nn.Dropout(p = dropout)
        self.conv1 = nn.Conv1d(in_channels = 1, out_channels = conv1_out, kernel_size = kernel1)
        self.conv2 = nn.Conv1d(in_channels = conv1_out, out_channels = conv2_out, kernel_size = kernel2)
        self.outlen1 = (timesteps - kernel1) + 1
        self.outlen2 = (self.outlen1 - kernel2) + 1
        self.gru = nn.GRU(input_size = conv2_out, hidden_size = hidden_size, num_layers = num_layers, batch_first = True)
        self.fc = nn.Linear(self.outlen2, 1)
    def forward(self, x):
        print('x.shape', x.shape)
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.gru(x)
        print('x.shape', x.shape)
        x = torch.tensor(x)
        output = self.fc(x)
        return output

my_model = Net(timesteps).to(device)

my_model.train()
criterion = nn.L1Loss()
optimizer = optim.Adam(my_model.parameters(), lr = 0.001)
epochs = 200

train_losses = []
valid_losses = []
average_train_losses = []
average_valid_losses = []
early_stop_flag = 0
early_stopping_maxim_increase = 10
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
        valid_losses.append(loss.item())
    train_loss = np.average(train_losses)
    valid_loss = np.average(valid_losses)
    if epoch == 0:
      pass
    else:
      if valid_loss > average_valid_losses[-1]:
        early_stop_flag += 1
      else:
        early_stop_flag = 0
    if early_stop_flag > early_stopping_maxim_increase:
      last_epoch = epoch - 1
      break
    else:
      last_epoch = epoch
    average_train_losses.append(train_loss)
    average_valid_losses.append(valid_loss)
    print('epoch', epoch, 'train_loss', train_loss, 'valid_loss', valid_loss)
    train_losses = []
    valid_losses = []

# visualize the loss as the network trained
fig = plt.figure(figsize=(10,8))
plt.plot(range(last_epoch + 1),average_train_losses, label='Training Loss')
plt.plot(range(last_epoch + 1),average_valid_losses,label='Validation Loss')

# find position of lowest validation loss
minposs = average_valid_losses.index(min(average_valid_losses)) 
plt.axvline(minposs, linestyle='--', color='r',label='Minimum validation loss')

plt.xlabel('epochs')
plt.ylabel('loss')
plt.ylim(0, 0.15) # consistent scale
plt.xlim(0, len(average_train_losses)+1) # consistent scale
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
fig.savefig('loss_plot_cnn-gru.png', bbox_inches='tight')

my_model.eval()
for sequence, label in test_loader:
    sequence = sequence.to(device)
    label = label.to(device)
    output = my_model(sequence)
    loss = criterion(output, label)
    print('MAE loss:',loss.item())