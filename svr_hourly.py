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
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib as mpl
import sklearn
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_percentage_error as MAPE
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
timesteps = 24
train_hours = 180 * 24 #day*24h
test_hours = 30 * 24 #day*24h

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

train_list, test_list = train_test(hourly_consumption_list, train_hours, test_hours)
train_data_timesteps = np.array([[j for j in train_list[i:i+timesteps]] for i in range(0,len(train_list)-timesteps+1)])
test_data_timesteps = np.array([[j for j in test_list[i:i+timesteps]] for i in range(0,len(test_list)-timesteps+1)])
x_train, y_train = train_data_timesteps[:,:timesteps-1],train_data_timesteps[:,[timesteps-1]]
x_test, y_test = test_data_timesteps[:,:timesteps-1],test_data_timesteps[:,[timesteps-1]]


#hyper parameters
kernel = 'rbf'
gamma = 0.5
C = 10
epsilon = 0.05

my_model = SVR(kernel = kernel, gamma = gamma, C = C, epsilon = epsilon)
start_time = time.time()
my_model.fit(x_train, y_train[:, 0])
time_fit = time.time() - start_time

y_train_pred = my_model.predict(x_train).reshape(-1,1)
start_time = time.time()
y_test_pred = my_model.predict(x_test).reshape(-1,1)
inferrence_time_per_one = (time.time() - start_time)/len(x_test)

# print('MAE fo training data: ', mean_absolute_error(y_train_pred, y_train))
# print('MAE fo testing data: ', mean_absolute_error(y_test_pred, y_test))
MAE_loss = MAE(y_test, y_test_pred)
MSE_loss = MSE(y_test, y_test_pred)
RMSE_loss = MSE(y_test, y_test_pred, squared = False)
MAPE_loss = MAPE(y_test, y_test_pred)
print('Total fitting time: ', time_fit, '\n', 'MAE: ', MAE_loss, '\n', 'MSE: ', MSE_loss, '\n', 'RMSE: ', RMSE_loss, '\n', 'MAPE: ', MAPE_loss, 'Inferrence time per one: ', inferrence_time_per_one)
with open('SVR_hourly_results.txt', 'a') as f:
    print('=====Testing results=====', file = f)
    print('Total fitting time: ', time_fit, '\n', 'MAE: ', MAE_loss, '\n', 'MSE: ', MSE_loss, '\n', 'RMSE: ', RMSE_loss, '\n', 'MAPE: ', MAPE_loss, '\n', 'Inferrence time per one: ', inferrence_time_per_one, file = f)

# plt.plot(range(len(y_test)), y_test)
# plt.plot(range(len(y_test_pred)), y_test_pred)
plt.plot(range(500), y_test[0:500])
plt.plot(range(500), y_test_pred[0:500])
plt.show()