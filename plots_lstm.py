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

#hypter parameters
train_hours = 180 * 24 #data for training
test_hours = 30 * 24
valid_hours = 30 * 24
sequence_length = 24 #forecast from past these hours
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
  print(len(source_list))
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


test_set = Dataset(input_sequences_test, label_test)
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
    output = self.fc(final_hidden_state[-1].view(len(final_hidden_state[-1]), -1))
    return output


my_model = Net(input_size, hidden_size).to('cpu')
my_model.load_state_dict(torch.load('./best-model-parameters-lstm-accumulate.pt'), torch.device('cpu'))
my_model.eval()

test_output = []
test_label = []
for sequence, label in test_loader_graph:
    output = my_model(sequence)
    output = output.reshape(1)
    label = label.reshape(1)
    test_label.append(label.detach().numpy())
    test_output.append(output.detach().numpy())
# print(test_output)
# print(test_label)
# plt.plot(range(0, len(test_output)), test_output)
# plt.plot(range(0, len(test_label)), test_label)
fig, ax = plt.subplots()
plt.plot(range(100), test_label[-101:-1], label = 'Ground truth')
plt.plot(range(100), test_output[-101:-1], label = 'LSTM')
leg = ax.legend()
plt.show()