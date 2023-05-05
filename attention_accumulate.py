#Accumulated data
#Use parameters of when validation dataset scored the best for test.
#MAE, MSE, RMSE, training time, inferrence time
#graph for test results
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
#!!!reference_past_hours should be shorter than train_hours, valid_hours and test_hours!!!
vector_length = 4
reference_past_hours = 7 * 24 #day*24h
train_hours = 180 * 24 #day*24h
valid_hours = 30 * 24 #day*24h
test_hours = 30 * 24 #day*24h
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

train_list, test_valid_list = train_test(accumulated_consumption_list, train_hours, test_hours + valid_hours)
test_list, valid_list = train_test(test_valid_list, test_hours, valid_hours)
key_train, query_train, value_train, label_train = key_query_value_label(train_list, reference_past_hours)
key_test, query_test, value_test, label_test = key_query_value_label(test_list, reference_past_hours)
key_valid, query_valid, value_valid, label_valid = key_query_value_label(valid_list, reference_past_hours)

class Dataset(nn.Module):
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

training_set = Dataset(key_train, query_train, value_train, label_train)
train_loader = DataLoader(training_set, batch_size = batch_size, drop_last = True)
valid_set = Dataset(key_valid, query_valid, value_valid, label_valid)
valid_loader = DataLoader(valid_set, batch_size = len(valid_set)) 
test_set = Dataset(key_test, query_test, value_test, label_test)
test_loader = DataLoader(test_set, batch_size = len(test_set))
test_loader_graph = DataLoader(test_set, batch_size = 1)

#hyper paramers
num_heads = 8
embed_dim = 80
fc_param1 = 64 #should be less than embed_dim
fc_param2 = 32
class Net(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(vector_length, embed_dim)
        self.fc2 = nn.Linear(1, embed_dim)
        self.fc3 = nn.Linear(embed_dim, 1)
        # self.fc4 = nn.Linear(fc_param1, fc_param2)
        # self.fc5 = nn.Linear(fc_param2, 1)
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first = True)
    def forward(self, key, query, value):
        key = F.relu(self.fc1(key))
        query = F.relu(self.fc1(query))
        value = F.relu(self.fc2(value))
        attn_output, attn_output_weights = self.multihead_attn(query, key, value)
        output = self.fc3(attn_output)
        # x = self.fc4(x)
        # output = self.fc5(x)
        return output

my_model = Net(embed_dim, num_heads).to(device)
#hyper paramers
criterion = nn.L1Loss()
optimizer = optim.Adam(my_model.parameters(), lr = 0.001)
# lambda1 = lambda epoch: 0.1 ** epoch
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
epochs = 300
patience = 0
min_patience = 300

train_losses = []
valid_losses = []
average_train_losses = []
average_valid_losses = []
start_time = time.time()
for epoch in range(epochs):
  #=============train==================
    my_model.train()
    for key, query, value, label in train_loader:
        key = key.to(device)
        query = query.to(device)
        value = value.to(device)
        label = label.to(device)
        output = my_model(key, query, value)
        optimizer.zero_grad()
        loss = criterion(output, label).to(torch.float32)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
  #=============valid==================
    my_model.eval()
    for key, query, value, label in valid_loader:
        key = key.to(device)
        query = query.to(device)
        value = value.to(device)
        label = label.to(device)
        output = my_model(key, query, value)
        loss = criterion(output, label).to(torch.float32)
        valid_loss = loss.item()
    train_loss = np.average(train_losses)
    if epoch == 0:
      best_valid_loss = valid_loss
      running_patience = 0
      best_epoch = 0
      torch.save(my_model.to('cpu').state_dict(),  'best-model-parameters-attention-accumulate.pt')
      my_model.to(device)
    else:
      if valid_loss < best_valid_loss:
        running_patience = 0
        best_valid_loss = valid_loss
        best_epoch = epoch
        torch.save(my_model.to('cpu').state_dict(),  'best-model-parameters-attention-accumulate.pt')
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
with open('attention_accum_results.txt', 'a') as f:
    print('===============================================================', file = f)
    print('=====Training results=====', '\n', 'Best valid loss: ', best_valid_loss, '\n', 'Best epoch: ', best_epoch, '\n', 'Total time for training: ', total_training_time, '\n', 'Time per epoch', time_per_epoch, file = f)

criterion_MAE = nn.L1Loss()
criterion_MSE = nn.MSELoss()
criterion_MAPE = MAPE()
my_model.load_state_dict(torch.load('./best-model-parameters-attention-accumulate.pt'))
my_model.eval()
my_model.to('cpu')
for key, query, value, label in test_loader:
    output = my_model(key, query, value)
    loss_MAE = criterion_MAE(output, label)
    loss_MSE = criterion_MSE(output, label)
    loss_RMSE = torch.sqrt(criterion_MSE(output, label))
    label_reshape = label
    loss_MAPE = criterion_MAPE(output, label)
    print('=====Test results=====')
    print('MAE loss:',loss_MAE.item())
    print('MSE loss:',loss_MSE.item())
    print('RMSE loss:',loss_RMSE)
    print('MAPE: ', loss_MAPE)
with open('attention_accum_results.txt', 'a') as f:
    print('=====Test results=====', '\n', 'MAE loss: ', loss_MAE.item(), '\n', 'MSE loss:',loss_MSE.item(), '\n', 'RMSE loss:',loss_RMSE, '\n', 'Time per epoch', time_per_epoch, 'MAPE: ', loss_MAPE, file = f)

# ##Inference time on GPU
# my_model.to(device)
# start_time = time.time()
# for key, query, value, label in test_loader_graph:
#     key = key.to(device)
#     query = query.to(device)
#     value = value.to(device)
#     output = my_model(key, query, value)
# total_inferrence_time = time.time() - start_time
# inferrence_time_per_one = total_inferrence_time / len(test_loader_graph)
# print('Time for inferrence: ', inferrence_time_per_one)
# with open('attention_accum_results.txt', 'a') as f:
#     print('=====Inferrence=====', file = f)
#     print('Time for inferrence: ', inferrence_time_per_one, file = f)

# Inference time on CPU
start_time = time.time()
for key, query, value, label in test_loader_graph:
    output = my_model(key, query, value)
total_inferrence_time = time.time() - start_time
inferrence_time_per_one = total_inferrence_time / len(test_loader_graph)
print('Time for inferrence: ', inferrence_time_per_one)
with open('attention_accum_results.txt', 'a') as f:
    print('=====Inferrence=====', file = f)
    print('Time for inferrence: ', inferrence_time_per_one, file = f)