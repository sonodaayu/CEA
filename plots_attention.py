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
plt.plot(range(100), test_label[-101:-1], label = 'Ground truth')
plt.plot(range(100), test_output_attention[-101:-1], label = 'PVS-Attention')
leg = ax.legend()
plt.show()