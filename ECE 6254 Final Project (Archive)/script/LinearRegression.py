from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# Preprocess data
all_data = pd.read_csv('double_pendulum_dataset.csv')
all_data = all_data.values   # Column 0: theta1 / 1:theta2 / 2:time / 3: P1_x / 4:P1_y / 5:P2_x / 6:P2_y

index = [2, 3, 4, 5]
X = all_data[:, index]   # extract input feature
y = all_data[:, 6]  # Target output
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
poly = PolynomialFeatures(9)  # Choose feature size carefully, or it will leads to overfitting
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.fit_transform(X_test)
print(f'X_train dataset size:{X_train_poly.shape}')
print(f'y_train dataset size:{y_train.shape}')
print(f'X_test dataset size:{X_test_poly.shape}')
print(f'y_test dataset size:{y_test.shape}')

reg_poly = LinearRegression()
reg_poly.fit(X_train_poly, y_train)
R2 = reg_poly.score(X_test_poly, y_test)
print(f'The R2 score is {R2}')


# Performance demo
slice = 500
test_theta = all_data[1, 0:2]
test_predict = reg_poly.predict(X_test_poly)
test_predict = test_predict[0:slice]  
test_target = y_test[0:slice]
test_time = X_test[0:slice, 0]

train_predict = reg_poly.predict(X_train_poly)
train_target = y_train
train_time = X_train[:, 0]

# Plot train and test performance
plt.figure()

plt.subplot(211)
plt.xlabel('time (s)')
plt.ylabel('y position (m)')
plt.scatter(test_time, test_target, label='Test Target Output')
plt.scatter(test_time, test_predict, label='Test Predict Output')
plt.legend()
plt.grid()
plt.title(f'Test Performance')

plt.subplot(212)
plt.xlabel('time (s)')
plt.ylabel('y position (m)')
plt.scatter(train_time, train_predict, label='Train Target Output')
plt.scatter(train_time, train_target, label='Train Predict Output')
plt.legend()
plt.grid()
plt.title(f'Train Performance')
plt.show()
