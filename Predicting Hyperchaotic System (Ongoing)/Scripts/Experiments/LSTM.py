import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class chaosLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super(chaosLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.reg = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x, _ = self.rnn(x) # (seq, batch, hidden)
        s, b, h = x.shape
        x = x.view(s*b, h) # 转换成线性层的输入格式
        x = self.reg(x)
        x = x.view(s, b, -1)
        return x
    
def create_dataset(dataset, look_back=1):
    '''
    @Params: dataset: Train and test dataset ; look_back: predict time horizon
    @Description: Create dataset for training LSTM network
    '''
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        dataX.append(dataset[i:(i + look_back)])
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

# Read csv data and preprocess
dataframe = read_csv('3D-MLGM_dataset.csv', usecols=[0])  # get X data
dataset = dataframe.values
dataset = dataset.astype('float32')
dataset = dataset[:5000]

scaler = MinMaxScaler()
dataset = scaler.fit_transform(dataset)
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
trainX, trainY = create_dataset(train)
testX, testY = create_dataset(test)

trainX = trainX.reshape(-1, 1, trainX.shape[1])
trainY = trainY.reshape(-1, 1, trainY.shape[1])
testX = testX.reshape(-1, 1, testX.shape[1])

train_x = torch.from_numpy(trainX)
train_y = torch.from_numpy(trainY)
test_x = torch.from_numpy(testX)

print(f'trainX size = {trainX.shape}, trainY size = {trainY.shape}, testX szie = {testX.shape}')

# Training Params
net = chaosLSTM(input_size=1, hidden_size=16, output_size=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
epochs = 100

for e in range(epochs):
    var_x = Variable(train_x)
    var_y = Variable(train_y)
    
    out = net(var_x)
    loss = criterion(out, var_y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (e + 1) % 10 == 0:
        print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.data))

net = net.eval()
train_pred_y = net(train_x)
train_pred_y = train_pred_y.data.numpy()
train_y = train_y.data.numpy()
train_pred_y = train_pred_y.reshape(-1, 1)
train_y = train_y.reshape(-1, 1)
train_time = np.arange(train_y.shape[0])

train_pred_y = scaler.inverse_transform(train_pred_y)
train_y = scaler.inverse_transform(train_y)

plt.figure()
plt.xlabel('time step')
plt.ylabel('x')
plt.scatter(train_time, train_y, label='Real')
plt.scatter(train_time, train_pred_y, label='Predict')
plt.legend()
plt.grid()
plt.title(f'Train Performance')
plt.show()