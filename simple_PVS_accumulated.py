import pandas as pd
import numpy as np
from numpy.linalg import norm
from statistics import mean
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
#!!!reference_past_hours should be shorter than train_hours, valid_hours and test_hours!!!
vector_length = 24
train_pool_hours = 180 * 24 #day*24h
test_pool_hours = 30 * 24 #day*24h
num_sim_vectors = 5 #Num_sim_vectors most similar vectors are adopted for forecasting

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

train_list, test_list = train_test(accumulated_consumption_list, train_pool_hours, test_pool_hours)

def make_vectors(accumulated_consumption_list):
    num_vectors = len(accumulated_consumption_list) - vector_length
    vectors = []
    for i in range(num_vectors):
        vector = []
        for j in range(vector_length):
            vector.append(accumulated_consumption_list[i + j])
        vectors.append(vector)
    vectors = np.array(vectors)
    labels = []
    for i in range(num_vectors):
        labels.append(accumulated_consumption_list[i + vector_length])
    return vectors, labels


train_pool_vectors, train_pool_labels = make_vectors(train_list)
test_pool_vectors, test_pool_labels = make_vectors(test_list) 

forecasted_values = []
start_time = time.time()
for test_vector in test_pool_vectors:
    cosine_similarity_list = []
    for train_vector in train_pool_vectors:
        cosine_similarity = np.dot(train_vector, test_vector)/(norm(train_vector)*norm(test_vector))
        cosine_similarity_list.append(cosine_similarity)
    index_list = sorted(range(len(cosine_similarity_list)), key = lambda sub: cosine_similarity_list[sub])[-num_sim_vectors:]
    labels_similar = []
    for index in index_list:
        labels_similar.append(train_pool_labels[index])
    forecasted_value = mean(labels_similar)
    forecasted_values.append(forecasted_value)
inference_time_per_one = (time.time() - start_time)/len(forecasted_values)

# print('MAE fo training data: ', mean_absolute_error(y_train_pred, y_train))
# print('MAE fo testing data: ', mean_absolute_error(y_test_pred, y_test))
MAE_loss = MAE(test_pool_labels, forecasted_values)
MSE_loss = MSE(test_pool_labels, forecasted_values)
RMSE_loss = MSE(test_pool_labels, forecasted_values, squared = False)
MAPE_loss = MAPE(test_pool_labels, forecasted_values)
print('MAE: ', MAE_loss, '\n', 'MSE: ', MSE_loss, '\n', 'RMSE: ', RMSE_loss, '\n', 'MAPE: ', MAPE_loss, 'Inferrence time per one: ', inference_time_per_one)
with open('simple_PVS_accumulated_results.txt', 'a') as f:
    print('=====Testing results=====', file = f)
    print('MAE: ', MAE_loss, '\n', 'MSE: ', MSE_loss, '\n', 'RMSE: ', RMSE_loss, '\n', 'MAPE: ', MAPE_loss, '\n', 'Inferrence time per one: ', inference_time_per_one, file = f)

# plt.plot(range(len(y_test)), y_test)
# plt.plot(range(len(y_test_pred)), y_test_pred)
plt.plot(range(100), test_pool_labels[0:100])
plt.plot(range(100), forecasted_values[0:100])
plt.show()