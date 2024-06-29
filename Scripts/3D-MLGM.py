'''
Description: This code is used for generating time series dataset of Discrete Memristive Hyperchaotic Systems (MHS)
            Dataset: Time| States
'''

import numpy as np
from scipy.integrate import solve_ivp, odeint
import pandas as pd
import matplotlib.pyplot as plt

k = np.array([0.1, 3, 4, 0.1, 0.1])   # k0 = 0.1, k1 = 200, k2 in [-200, -0.01], [0.01, 200], k3 = 0.1, k4 = 0.1
initial_states = np.array([0.1, 0.1, 0.1])
a = 0.1
b = 1
t = 40000

x = np.zeros(t)
y = np.zeros(t)
q = np.zeros(t)
for n in range(t-1):
    x[0], y[0], q[0] = initial_states
    x[n+1] = k[1]*np.cos(y[n]*(1-y[n]))
    y[n+1] = k[2]*(np.exp(a*x[n]*x[n])-b) + k[0]*np.cos(q[n])*y[n]
    q[n+1] = k[3]*y[n] + k[4]*q[n]

data = np.vstack((x, y, q))
save_data = data.T
extracted_data = data[:, ::10]
show_data = extracted_data

# Show in fig
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.plot(show_data[0, 1000:], show_data[1, 1000:], show_data[2, 1000:])
ax.scatter(show_data[0, 1000:], show_data[1, 1000:], show_data[2, 1000:], c='b', marker='o')

# Used for plotting asymptotic color change
# ax.scatter(show_data[0, 1000:2000], show_data[1, 1000:2000], show_data[2, 1000:2000], c='r', marker='o')
# ax.scatter(show_data[0, 2000:], show_data[1, 2000:], show_data[2, 2000:], c='b', marker='o')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('q')
plt.show()

# Save to CSV
df = pd.DataFrame(save_data, columns=['x', 'y', 'q'])
csv_path = '3D-MLCM_dataset.csv'
df.to_csv(csv_path, index=False)

