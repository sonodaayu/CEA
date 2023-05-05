# from anyio import current_effective_deadline
import pandas as pd
import numpy as np
import time
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
    device = torch.device("cuda:0")
else:
    print('no GPU available')
    device = torch.device("cpu")

df = pd.read_csv('../DATA/edf_data_hard.csv')

#hyper parameters
train_hours = 180 * 24 #data for training
test_hours = 30 * 24
valid_hours = 30 * 24
sequence_length = 7 * 24 #forecast from past these hours
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


def train_test(accumulated_consumption_list, train_hours, test_hours):
    train_list = accumulated_consumption_list[:train_hours]
    test_list = accumulated_consumption_list[train_hours:train_hours+test_hours]
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

train_list, test_valid_list = train_test(accumulated_consumption_list, train_hours, test_hours + valid_hours)
test_list, valid_list = train_test(test_valid_list, test_hours, valid_hours)
input_sequences_train, label_train = source_list_to_input_sequences(train_list)
input_sequences_test, label_test = source_list_to_input_sequences(test_list)
input_sequences_valid, label_valid = source_list_to_input_sequences(valid_list)

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
test_set = Dataset(input_sequences_test, label_test)
valid_set = Dataset(input_sequences_valid, label_valid)
train_loader = DataLoader(training_set, batch_size = batch_size, drop_last = True)
valid_loader = DataLoader(valid_set, batch_size = len(valid_set))
test_loader = DataLoader(test_set, batch_size = test_hours)
test_loader_graph = DataLoader(test_set, batch_size = 1)

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
  #=============valid=================
  my_model.eval()
  for input_sequence, label in valid_loader:
      input_sequence = input_sequence.to(device)
      label = label.to(device)
      output = my_model(input_sequence)
      loss = criterion(output, label).to(torch.float32)
      valid_loss = loss.item()
  train_loss = np.average(train_losses)
  if epoch == 0:
      best_valid_loss = valid_loss
      running_patience = 0
      best_epoch = 0
      torch.save(my_model.to('cpu').state_dict(),  'best-model-parameters-lstm-accumulate.pt')
      my_model.to(device)
  else:
      if valid_loss < best_valid_loss:
          running_patience = 0
          best_valid_loss = valid_loss
          best_epoch = epoch
          torch.save(my_model.to('cpu').state_dict(),  'best-model-parameters-lstm-accumulate.pt')
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
with open('lstm_accumulated_results.txt', 'a') as f:
    print('===============================================================', file = f)
    print('Train hours: ', train_hours, 'Valid hours: ', valid_hours, 'Test hours: ', test_hours, 'Batch size: ', batch_size, 'Input size: ', input_size, 'Hidden size: ', hidden_size, 'Output size: ', output_size, 'Criterion: ', my_model.__class__.__name__,)
    print('=====Training results=====', '\n', 'Best valid loss: ', best_valid_loss, '\n', 'Best epoch: ', best_epoch, '\n', 'Total time for training: ', total_training_time, '\n', 'Time per epoch', time_per_epoch, file = f)

criterion_MAE = nn.L1Loss()
criterion_MSE = nn.MSELoss()
criterion_MAPE = MAPE()
my_model.load_state_dict(torch.load('./best-model-parameters-lstm-accumulate.pt'))
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
with open('lstm_accumulated_results.txt', 'a') as f:
    print('=====Test results=====', '\n', 'MAE loss: ', loss_MAE.item(), '\n', 'MSE loss:',loss_MSE.item(), '\n', 'RMSE loss:',loss_RMSE, '\n', 'Time per epoch', time_per_epoch, 'MAPE: ', loss_MAPE, file = f)


# Inference on GPU
my_model = my_model.to(device)
start_time = time.time()
for sequence, label in test_loader_graph:
    sequence = sequence.to(device)
    output = my_model(sequence)
total_inferrence_time = time.time() - start_time
inferrence_time_per_one = total_inferrence_time / len(test_loader_graph)
print('Time for inferrence: ', inferrence_time_per_one)
with open('lstm_accumulated_results.txt', 'a') as f:
    print('=====Inferrence=====', file = f)
    print('Time for inferrence: ', inferrence_time_per_one, file = f)



## Inference on CPU
# start_time = time.time()
# for sequence, label in test_loader_graph:
#     output = my_model(sequence)
# total_inferrence_time = time.time() - start_time
# inferrence_time_per_one = total_inferrence_time / len(test_loader_graph)
# print('Time for inferrence: ', inferrence_time_per_one)
# with open('lstm_accumulated_results.txt', 'a') as f:
#     print('=====Inferrence=====', file = f)
#     print('Time for inferrence: ', inferrence_time_per_one, file = f)