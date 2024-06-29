'''
Description: This code is used for generating time series dataset of Discrete Memristive Hyperchaotic Systems (MHS)
            Dataset: Time| States
'''

import numpy as np
from scipy.integrate import solve_ivp, odeint
import pandas as pd
import matplotlib.pyplot as plt

k = np.array([0.1, 3, 4, 0.1, 0.1])     # k0 = 0.1, k1 = 200, k2 in [-200, -0.01], [0.01, 200], k3 = 0.1, k4 = 0.1
initial_states = np.array([0.1, 0.1, 0.1])
a = 0.1
b = 1
# def MHS(state, t):
def MHS(t, state):
    '''
    @params: state: x_n, y_n, q_n  t: time steps  k: coefficient
    @return: dstate: x_n+1, y_n+1, q_n+1
    @description: The system dynamics of 3-D memristive logistic-cubic map (3D-MLGM)
    '''
    dstate = np.zeros_like(state)
    x, y, q = state

    dstate[0] = k[1] * np.cos(y*(1-y))
    # dstate[1] = k[2] * (x*(1-x**2)) + k[0] * np.cos(q)*y  # 3D-MLCM
    dstate[1] = k[2] * (np.exp(a*x*x)-b) + k[0] * np.cos(q) * y
    dstate[2] = k[3] * y + k[4] * q

    dstate = dstate

    return dstate

def generate_dataset(initial_states, t, dt):
    '''
    @params: initial_states: x_0, y_0, q_0  t: time steps  dt: time gap
    @return: data: t/dt x 4 dataset
    @description: 3D-MLGM system dataset (using RK45)
    '''
    tspan = (0, t)
    sys_output = solve_ivp(MHS, tspan, initial_states)

    return sys_output

output = generate_dataset(initial_states, 20, 0.001)
data = output.y
data = data.T

# Save to CSV
df = pd.DataFrame(data, columns=['x', 'y', 'q'])
csv_path = '3D-MLCM_dataset.csv'
df.to_csv(csv_path, index=False)


# Show in fig
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.plot(data[:, 0], data[:, 1], data[:, 2])
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='b', marker='o')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('q')
plt.show()