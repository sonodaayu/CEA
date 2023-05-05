#Accumulated data
#Use parameters of when validation dataset scored the best for test.
#MAE, MSE, RMSE, training time, inferrence time
#graph for test results
import time
import pandas as pd
import numpy as np
import random
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

#hyper parameters
#!!!reference_past_hours should be shorter than train_hours, valid_hours and test_hours!!!
vector_length = 4
reference_past_hours = 7 * 24 #day*24h
train_hours = 180 * 24 #day*24h
valid_hours = 30 * 24 #day*24h
test_hours = 30 * 24 #day*24h
batch_size = 64

train_list = [random.choice(range(1, 10)) for _ in range(180*24)]
test_list = [random.choice(range(1, 10)) for _ in range(30*24)]

#normalization
train_list = sklearn.preprocessing.minmax_scale(train_list)
consumption_value_list = train_list.tolist()
test_list = sklearn.preprocessing.minmax_scale(test_list)
test_list = test_list.tolist()
# print(hourly_consumption_list)

def key_query_value_label(consumption_value_list, reference_past_hours):
    key = []
    # if num_data  > len(hourly_consumption_list) - vector_length - reference_past_hours:
    #   print('Data insufficient. Need to either shorten the reference_past_vectors or num_data')
    #   return 0
    num_data = len(consumption_value_list) - reference_past_hours - vector_length - 1
    for i in range(num_data):
        vectors_in_key = []
        for j in range(reference_past_hours):
            vector_unit = []
            for k in range(vector_length):
                vector_unit.append(consumption_value_list[i + j + k])
            vectors_in_key.append(vector_unit)
        key.append(vectors_in_key)
    query = []
    for i in range(num_data):
      query_unit = []
      for j in range(vector_length):
        query_unit.append(consumption_value_list[i + j + reference_past_hours])
      query.append(query_unit)
    value = []
    for i in range(num_data):
      value_unit = []
      for j in range(reference_past_hours):
        value_unit.append(consumption_value_list[i + j +  vector_length])
      value.append(value_unit)
    label = []
    for i in range(num_data):
        label.append(consumption_value_list[i + reference_past_hours + vector_length])

    key = np.array(key)
    query = np.array(query)
    query = query.reshape(num_data, 1, vector_length)
    value = np.array(value)
    value = value.reshape(num_data, reference_past_hours, 1)
    label = np.array(label)
    label = label.reshape(num_data, 1, 1)
    return key, query, value, label

key_train, query_train, value_train, label_train = key_query_value_label(train_list, reference_past_hours)
key_test, query_test, value_test, label_test = key_query_value_label(test_list, reference_past_hours)

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
test_set = Dataset(key_test, query_test, value_test, label_test)
test_loader = DataLoader(test_set, batch_size = 1)

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
epochs = 1
patience = 0
min_patience = 1

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
    train_loss = np.average(train_losses)
    average_train_losses.append(train_loss)
    print('epoch', epoch, 'train_loss', train_loss)
    train_losses = []

my_model.eval()
start_time = time.time()
for key, query, value, label in test_loader:
    output = my_model(key, query, value)
total_inferrence_time = time.time() - start_time
inferrence_time_per_one = total_inferrence_time / len(test_loader)
print('Time for inferrence: ', inferrence_time_per_one)
with open('attention.txt', 'a') as f:
    print('Time for inferrence: ', inferrence_time_per_one, file = f)

