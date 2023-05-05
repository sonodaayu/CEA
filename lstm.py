import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.DataLoader as DataLoader
import datetime as dt

import matplotlib.pyplot as plt
import matplotlib as mpl

import sklearn
from sklearn.preprocessing import MinMaxScaler

if torch.cuda.is_available():
    print(torch.cuda.device_count())
    print('Training runs on GPU')
    device = torch.device("cuda:0"h)
else:
    print('no GPU available')
    device = torch.device("cpu")


print('<<<<<<<<<<< Data >>>>>>>>>>>')
df = pd.read_csv('../DATA/edf_data_easy.csv')
print(df.head(5))
df = df[['V40']]


def train_test(df, test_periods):
    train = df[:-test_periods].values
    test = df[-test_periods:].values
    return train, test
test_periods = 8
train, test = train_test(df, test_periods)

scaler = MinMaxScaler()
scaler.fit(train)
train_scaled = scaler.transform(train)
train_scaled = torch.FloatTensor(train_scaled)
train_scaled = train_scaled.reshape(-1)

def get_x_y_pairs(train_scaled, train_periods, prediction_periods):
    """
    train_scaled - training sequence
    train_periods - How many data points to use as inputs
    prediction_periods - How many periods to ouput as predictions
    """
    x_train = [train_scaled[i:i+train_periods] for i in range(len(train_scaled)-train_periods-prediction_periods)]
    y_train = [train_scaled[i+train_periods:i+train_periods+prediction_periods] for i in range(len(train_scaled)-train_periods-prediction_periods)]
    x_train = torch.stack(x_train)
    y_train = torch.stack(y_train)
    return x_train, y_train




train_periods = 120 #-- number of quarters for input
prediction_periods = test_periods
x_train, y_train = get_x_y_pairs(train_scaled, train_periods, prediction_periods)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden=None):
        if hidden==None:
            self.hidden = (torch.zeros(1,1,self.hidden_size, device = x.device),
                           torch.zeros(1,1,self.hidden_size, device = x.device))
        else:
            self.hidden = hidden
        lstm_out, self.hidden = self.lstm(x.view(len(x),1,-1), 
                                          self.hidden)
        predictions = self.linear(lstm_out.view(len(x), -1))
        return predictions[-1], self.hidden

model = LSTM(input_size=1, hidden_size=50, output_size=test_periods, batch_first = True).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 600
model.train()
print(len(x_train))
for epoch in range(epochs+1):
    for x,y in zip(x_train, y_train):
        x = x.to(device)
        y = y.to(device)
        y_hat, _ = model(x, None)
        optimizer.zero_grad()
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        
    # if epoch%100==0:
    #     print(f'epoch: {epoch:4} loss:{loss.item():10.8f}')
    print(f'epoch: {epoch:4} loss:{loss.item():10.8f}')




model.eval()
with torch.no_grad():
    predictions, _ = model(train_scaled[-train_periods:], None)
#-- Apply inverse transform to undo scaling
predictions = scaler.inverse_transform(np.array(predictions.reshape(-1,1)))
print(predictions)




# x = [dt.datetime.date(d) for d in df.index]
# font = {'size'   : 15}

# mpl.rc('font', **font)
# fig = plt.figure(figsize=(10,5))
# plt.title('Walmart Quarterly Revenue')
# plt.ylabel('Revenue (Billions)')
# plt.grid(True)
# plt.plot(x[:-len(predictions)],
#          df.Value[:-len(predictions)],
#          "b-")
# plt.plot(x[-len(predictions):],
#          df.Value[-len(predictions):],
#          "b--",
#          label='True Values')
# plt.plot(x[-len(predictions):],
#          predictions,
#          "r-",
#          label='Predicted Values')
# plt.legend()
# plt.savefig('plot1', dpi=600)