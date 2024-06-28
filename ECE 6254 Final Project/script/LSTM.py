from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mse

from keras.models import Sequential
from keras.layers import LSTM, Dense


TF_ENABLE_ONEDNN_OPTS=0

def create_lstm_dataset(dataset, forward_time=1):
    """
    input: dataset: time series data , forward_time: predict length
    output: dataset for trainning and testing
    """
    size = len(dataset) - forward_time
    X_data = dataset[0:size, :]
    y_data = dataset[forward_time:len(dataset), :]
    return X_data, y_data

# Preprocess data
all_data = pd.read_csv('double_pendulum_dataset.csv')
all_data = all_data.values   # Column 0: theta1 / 1:theta2 / 2:time / 3: P1_x / 4:P1_y / 5:P2_x / 6:P2_y
all_data = all_data.astype('float32')

index_x = [5]
X = all_data[:, index_x]   # extract input feature
time = all_data[:, 2]

scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)  # Transform input to [0, 1]
train_size = int(len(X) * 0.67)
test_size = len(X)-train_size
train_data, test_data = X[0:train_size, :], X[train_size:, :]

forward_time = 1
X_train, y_train = create_lstm_dataset(train_data, forward_time=forward_time)
X_test, y_test = create_lstm_dataset(test_data, forward_time=forward_time)
train_time, test_time = time[0:train_size], time[train_size:]

print(f'X_train dataset size:{X_train.shape}')
print(f'y_train dataset size:{y_train.shape}')
print(f'X_test dataset size:{X_test.shape}')
print(f'y_test dataset size:{y_test.shape}')

X_train_tensor = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test_tensor = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# --------------trainning------------------
INPUT_FEATURES_NUM = X.shape[1]
OUTPUT_FEATURES_NUM = X.shape[1]
model = Sequential()
model.add(LSTM(4, input_shape=(1, INPUT_FEATURES_NUM)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_train_tensor, y_train, epochs=20, batch_size=1, verbose=2)

X_train_predict = model.predict(X_train_tensor)
X_test_predict = model.predict(X_test_tensor)

X_train_predict = scaler.inverse_transform(X_train_predict)
y_train = scaler.inverse_transform(y_train)
X_test_predict = scaler.inverse_transform(X_test_predict)
y_test = scaler.inverse_transform(y_test)

train_score = mse(X_train_predict, y_train)
test_score = mse(X_test_predict, y_test)

print(f'Train_score: MSE = {train_score}')
print(f'Test_score: MSE = {test_score}')

# Plot train and test performance
plt.figure()

plt.subplot(211)
plt.xlabel('time (s)')
plt.ylabel('y position (m)')
plt.scatter(test_time[0:test_size-forward_time], y_test, label='Test Target Output')
plt.scatter(test_time[forward_time:test_size], X_test_predict, label='Test Predict Output')
plt.legend()
plt.grid()
plt.title(f'Test Performance')

plt.subplot(212)
plt.xlabel('time (s)')
plt.ylabel('y position (m)')
plt.scatter(train_time[0:train_size-forward_time], y_train, label='Train Target Output')
plt.scatter(train_time[forward_time:train_size], X_train_predict, label='Train Predict Output')
plt.legend()
plt.grid()
plt.title(f'Train Performance')
plt.show()
