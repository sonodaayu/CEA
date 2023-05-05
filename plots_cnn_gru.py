#Accumulated data
#Use parameters of when validation dataset scored the best for test.
#MAE, MSE, RMSE, training time, inferrence time
#graph for test results
from statistics import mean
import time
import pandas as pd
import numpy as np
from numpy.linalg import norm
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
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
device = torch.device('cpu')
df = pd.read_csv('../DATA/edf_data_hard.csv')

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

train_hours = 180 * 24 #day*24h
valid_hours = 30 * 24 #day*24h
test_hours = 30 * 24 #day*24h
plt_from = 100
plt_end = 200

def train_test(accumulated_consumption_list, train_hours, test_hours):
    train_list = accumulated_consumption_list[:train_hours]
    test_list = accumulated_consumption_list[train_hours:train_hours+test_hours]
    return train_list, test_list

train_list, test_valid_list = train_test(accumulated_consumption_list, train_hours, test_hours + valid_hours)
test_list, valid_list = train_test(test_valid_list, test_hours, valid_hours)

#================================================================ PVS-Attention ================================================================
vector_length = 4
reference_past_hours = 7 * 24 #day*24h
batch_size = 64
def key_query_value_label(accumulated_consumption_list, reference_past_hours):
    key = []
    # if num_data  > len(hourly_consumption_list) - vector_length - reference_past_hours:
    #   print('Data insufficient. Need to either shorten the reference_past_vectors or num_data')
    #   return 0
    num_data = len(accumulated_consumption_list) - reference_past_hours - vector_length - 1
    for i in range(num_data):
        vectors_in_key = []
        for j in range(reference_past_hours):
            vector_unit = []
            for k in range(vector_length):
                vector_unit.append(accumulated_consumption_list[i + j + k])
            vectors_in_key.append(vector_unit)
        key.append(vectors_in_key)
    query = []
    for i in range(num_data):
      query_unit = []
      for j in range(vector_length):
        query_unit.append(accumulated_consumption_list[i + j + reference_past_hours])
      query.append(query_unit)
    value = []
    for i in range(num_data):
      value_unit = []
      for j in range(reference_past_hours):
        value_unit.append(accumulated_consumption_list[i + j +  vector_length])
      value.append(value_unit)
    label = []
    for i in range(num_data):
        label.append(accumulated_consumption_list[i + reference_past_hours + vector_length])

    key = np.array(key)
    query = np.array(query)
    query = query.reshape(num_data, 1, vector_length)
    value = np.array(value)
    value = value.reshape(num_data, reference_past_hours, 1)
    label = np.array(label)
    label = label.reshape(num_data, 1, 1)
    return key, query, value, label

class Dataset_attention(nn.Module):
    def __init__(self, key, query, value, label):
        self.key = key
        self.query = query
        self.value = value
        self.label = label
    def __len__(self):
        return len(self.value)
    def __getitem__(self, index):
        key = self.key[index]
        query = self.query[index]
        value = self.value[index]
        label = self.label[index]
        key = np.float32(key)
        query = np.float32(query)
        value = np.float32(value)
        label = np.float32(label)
        key = torch.from_numpy(key).requires_grad_(True)
        query = torch.from_numpy(query).requires_grad_(True)
        value = torch.from_numpy(value).requires_grad_(True)
        return key, query, value, label

key_test, query_test, value_test, label_test = key_query_value_label(test_list, reference_past_hours)
test_set_attention = Dataset_attention(key_test, query_test, value_test, label_test)
test_loader_graph_attention = DataLoader(test_set_attention, batch_size = 1)

#hyper paramers
num_heads = 8
embed_dim = 80
class Net_attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Net_attention, self).__init__()
        self.fc1 = nn.Linear(vector_length, embed_dim)
        self.fc2 = nn.Linear(1, embed_dim)
        self.fc3 = nn.Linear(embed_dim, 1)
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first = True)
    def forward(self, key, query, value):
        key = F.relu(self.fc1(key))
        query = F.relu(self.fc1(query))
        value = F.relu(self.fc2(value))
        attn_output, attn_output_weights = self.multihead_attn(query, key, value)
        output = self.fc3(attn_output)
        return output

my_model = Net_attention(embed_dim, num_heads).to(device)
my_model.load_state_dict(torch.load('./best-model-parameters-attention-accumulate.pt'), torch.device('cpu'))
my_model.eval()
test_output_attention = []
test_label= []
for key, query, value, label in test_loader_graph_attention:
    output = my_model(key, query, value)
    output = output.reshape(1)
    label = label.reshape(1)
    test_label.append(label.detach().numpy())
    test_output_attention.append(output.detach().numpy())
# print(test_output)
# print(test_label)
# plt.plot(range(0, len(test_output)), test_output)
# plt.plot(range(0, len(test_label)), test_label)
fig, ax = plt.subplots()
# plt.plot(range(100), test_output_attention[-101:-1], label = 'Attention')
plt.plot(range(100), test_label[-101:-1], label = 'Ground truth')
#================================================================ CNN-GRU ================================================================
timesteps = 7 * 24 #day*24h
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

class Dataset_cnn_gru(nn.Module):
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

test_inputs_cnn_gru, test_labels_cnn_gru = timestep_split(test_list)
test_set_cnn_gru = Dataset_cnn_gru(test_inputs_cnn_gru, test_labels_cnn_gru)
test_loader_graph_cnn_gru = DataLoader(test_set_cnn_gru, batch_size = 1)

#Hyper parameters
conv1_out = 50
conv2_out = 50
kernel1 = 16
kernel2 = 8
hidden_size = 50
num_layers = 2
dropout = 0.2
class Net_cnn_gru(nn.Module):
    def __init__(self, timesteps):
        super(Net_cnn_gru, self).__init__()
        self.dropout = nn.Dropout(p = dropout)
        self.conv1 = nn.Conv1d(in_channels = 1, out_channels = conv1_out, kernel_size = kernel1)
        self.conv2 = nn.Conv1d(in_channels = conv1_out, out_channels = conv2_out, kernel_size = kernel2)
        self.outlen1 = (timesteps - kernel1) + 1
        self.outlen2 = (self.outlen1 - kernel2) + 1
        self.gru = nn.GRU(input_size = conv2_out, hidden_size = hidden_size, num_layers = num_layers, batch_first = True)
        self.fc = nn.Linear(self.outlen2 * hidden_size, 1)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = x.reshape(x.size(0), x.size(-1), x.size(1))
        x, _ = self.gru(x)
        x = x.contiguous().view(x.size(0), -1) # make it to batch*num_nodes
        output = self.fc(x).squeeze(-1)
        return output


my_model = Net_cnn_gru(timesteps).to('cpu')
my_model.load_state_dict(torch.load('./best-model-parameters-cnn-gru-accumulate.pt'), torch.device('cpu'))
my_model.eval()
test_output_cnn_gru = []
test_labels_cnn_gru = []
for sequence, label in test_loader_graph_cnn_gru:
    output = my_model(sequence)
    output = output.reshape(1)
    label = label.reshape(1)
    test_labels_cnn_gru.append(label.detach().numpy())
    test_output_cnn_gru.append(output.detach().numpy())
# plt.plot(range(0, len(test_output)), test_output)
# plt.plot(range(0, len(test_label)), test_label)
plt.plot(range(100), test_output_cnn_gru[-101:-1], label = 'CNN-GRU')
leg = ax.legend()
plt.show()