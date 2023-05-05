from msilib import sequence
import pandas as pd
import numpy as np
from statistics import mean
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
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_absolute_error
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
forecast_ahead = 1
timesteps = 90 * 24  #day*24h
train_hours = 250 * 24 #day*24h
# test_hours = 30 * 24 #day*24h

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

def source_list_to_input_sequences(source_list):
  print(len(source_list))
  input_sequences = []
  for i in range(len(source_list) - timesteps):
    input_sequence = []
    for j in range(timesteps):
      input_sequence.append(source_list[i+j])
    input_sequences.append(input_sequence)
  
  label = []
  for i in range(len(source_list) - timesteps):
    label.append(source_list[i + timesteps])
  return input_sequences, label

sequences, labels = source_list_to_input_sequences(hourly_consumption_list)
mae_list = []
for i in range(len(labels)):
    my_model = auto_arima(sequences[i])
    prediction, confint = my_model.predict(n_periods = forecast_ahead, return_conf_int = True) # What is return_conf_int??
    print(type(prediction))
    prediction = prediction.astype(float)
    print(type(prediction))
    mae_list.append(abs(prediction - labels[i]))
    break

mean_mae = mean(mae_list)
print('MAE: ', mean_mae)