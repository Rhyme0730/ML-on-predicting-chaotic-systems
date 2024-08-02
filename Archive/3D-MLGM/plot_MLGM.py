import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

colors = ['#3682be','#f05326','#ddae33']

dataframe1 = pd.read_csv('k13.csv') 
k13 = dataframe1.values
k13 = k13.astype('float32')

dataframe2 = pd.read_csv('k14.csv')  
k14 = dataframe2.values[:3000, :]
k14 = k14.astype('float32')

dataframe3 = pd.read_csv('k15.csv')  
k15 = dataframe3.values[:3000, :]
k15 = k15.astype('float32')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(k13[:, 0], k13[:, 1], k13[:, 2], c=colors[0], marker='.', linewidths=0.1, label='k_1=3')
ax.scatter(k14[:, 0], k14[:, 1], k14[:, 2], c=colors[1], marker='.', linewidths=0.1, label='k_1=4')
ax.scatter(k15[:, 0], k15[:, 1], k15[:, 2], c=colors[2], marker='.', linewidths=0.1, label='k_1=5')
plt.grid()

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('q')
plt.legend()
plt.show()



# colors = ['#3682be','#f05326','#ddae33']

# dataframe1 = pd.read_csv('k23.csv') 
# k23 = dataframe1.values
# k23 = k23.astype('float32')

# dataframe2 = pd.read_csv('k24.csv')  
# k24 = dataframe2.values
# k24 = k24.astype('float32')

# dataframe3 = pd.read_csv('k25.csv')  
# k25 = dataframe3.values
# k25 = k25.astype('float32')

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(k23[:, 0], k23[:, 1], k23[:, 2], c='blue', marker='.', linewidths=0.1, label='k_2=3')
# ax.scatter(k24[:, 0], k24[:, 1], k24[:, 2], c='gold', marker='.', linewidths=0.1, label='k_2=4')
# ax.scatter(k25[:, 0], k25[:, 1], k25[:, 2], c='tomato', marker='.', linewidths=0.1, label='k_2=5')
# plt.grid()

# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('q')
# plt.legend()
# plt.show()







