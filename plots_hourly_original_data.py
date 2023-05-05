#Use parameters of when validation dataset scored the best for test.
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
plt.plot(range(100), hourly_consumption_list[0:100])
plt.show()
hourly_consumption_list = sklearn.preprocessing.minmax_scale(hourly_consumption_list)
hourly_consumption_list = hourly_consumption_list.tolist()
plt.plot(range(100), hourly_consumption_list[0:100])
plt.show()