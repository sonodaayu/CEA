# from anyio import current_effective_deadline
import pandas as pd
import numpy as np
import random
import time
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
    device = torch.device("cuda:0")
else:
    print('no GPU available')
    device = torch.device("cpu")


#hyper parameters
train_hours = 180 * 24 #data for training
test_hours = 30 * 24
valid_hours = 30 * 24
sequence_length = 7 * 24 #forecast from past these hours
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

def source_list_to_input_sequences(source_list):
  input_sequences = []
  for i in range(len(source_list) - sequence_length):
    input_sequence = []
    for j in range(sequence_length):
      input_sequence.append(source_list[i+j])
    input_sequences.append(input_sequence)
  
  label = []
  for i in range(len(source_list) - sequence_length):
    label.append(source_list[i + sequence_length])
  return input_sequences, label

train_list, test_valid_list = train_test(consumption_value_list, train_hours, test_hours + valid_hours)
input_sequences_train, label_train = source_list_to_input_sequences(train_list)


class Dataset(nn.Module):
    def __init__(self,input_sequences, label):
        self.input_sequences = input_sequences
        self.label = label
    def __len__(self):
        return len(self.label)
    def __getitem__(self, index):
        input_sequence = self.input_sequences[index]
        label = self.label[index]
        input_sequence = torch.FloatTensor(input_sequence)
        input_sequence = input_sequence.reshape(sequence_length, -1)
        return input_sequence, label

training_set = Dataset(input_sequences_train, label_train)
train_loader = DataLoader(training_set, batch_size = batch_size, drop_last = True)

#model
#hyper_parameters
input_size = 1 #num of the features of the input sequence
hidden_size = 50 #dimension of the hidden state
output_size = 1
class Net(nn.Module):
  def __init__(self, input_size, hidden_size):
    super(Net, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.lstm = nn.LSTM(input_size, hidden_size, batch_first = True)
    self.fc = nn.Linear(hidden_size, output_size)
  
  def forward(self, x, hidden = None):
    if hidden == None:
       h0 = torch.zeros(1, x.size(0), self.hidden_size, device = x.device, requires_grad = True)
       c0 = torch.zeros(1, x.size(0), self.hidden_size, device = x.device, requires_grad = True)
    else:
      self.hidden = hidden
    lstm_out, (final_hidden_state, final_cell_state) = self.lstm(x, (h0.detach(), c0.detach()))
    output = self.fc(final_hidden_state[-1].view(len(final_hidden_state[-1]), -1)).squeeze(-1)
    return output

my_model = Net(input_size, hidden_size).to(device)
#hyper parameters
criterion = nn.L1Loss()
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
  for input_sequence, label in train_loader:
    input_sequence = input_sequence.to(device)
    label = label.to(device)
    output = my_model(input_sequence)
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
print('Total training time: ', total_training_time, '\n', 'Time per epoch: ', time_per_epoch)
with open ('lstm.txt', 'a') as f:
    print('Total training time: ', total_training_time, 'Time per epoch "', time_per_epoch, file = f)
