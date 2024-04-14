from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import pandas as pd

# Double Pendulum params
G = 9.81  # acceleration due to gravity, in m/s^2
L1 = 1.0  # length of pendulum 1 in m
L2 = 1.0 # length of pendulum 2 in m
M1 = 1.0  # mass of pendulum 1 in kg
M2 = 1.0  # mass of pendulum 2 in kg

def double_Pendulum(state, t):
    '''
    :param state: state[0]:theta1  state[1]:theta1_dot  state[2]:theta2  state[3]:theta2_dot
           (all with respect to vertical)
    :param t: simulation time
    :return: dydx: derivative of state with respect to time
    '''
    noise = 0.0
    dydx = np.zeros_like(state)
    dydx[0] = state[1]
    del_ = state[2] - state[0] + noise
    den1 = (M1 + M2) * L1 - M2 * L1 * cos(del_) * cos(del_)
    dydx[1] = (M2 * L1 * state[1] * state[1] * sin(del_) * cos(del_) +
               M2 * G * sin(state[2]) * cos(del_) +
               M2 * L2 * state[3] * state[3] * sin(del_) -
               (M1 + M2) * G * sin(state[0])) / den1
    dydx[2] = state[3]
    den2 = (L2 / L1) * den1
    dydx[3] = (-M2 * L2 * state[3] * state[3] * sin(del_) * cos(del_) +
               (M1 + M2) * G * sin(state[0]) * cos(del_) -
               (M1 + M2) * L1 * state[1] * state[1] * sin(del_) -
               (M1 + M2) * G * sin(state[2])) / den2
    dydx = dydx
    return dydx

def generate_pendulum_dataset(th1, th2, t, dt):
  # System params
  G = 9.81  # acceleration due to gravity, in m/s^2
  L1 = 1.0  # length of pendulum 1 in m
  L2 = 1.0 # length of pendulum 2 in m
  M1 = 1.0  # mass of pendulum 1 in kg
  M2 = 1.0  # mass of pendulum 2 in kg

  tspan = np.arange(0.0, t, dt)  # Turn param 1: Time

  # Initial Conditions
  w1 = 0.0
  w2 = 0.0
  initial_state = np.radians([th1, w1, th2, w2])

  y = integrate.odeint(double_Pendulum, initial_state, tspan)

  P1 = np.dstack([L1 * sin(y[:, 0]), -L1 * cos(y[:, 0])]).squeeze()
  P2 = P1 + np.dstack([L2 * sin(y[:, 2]), -L2 * cos(y[:, 2])]).squeeze()

  data1 = np.array([th1*np.ones_like(y[:, 0]), th2*np.ones_like(y[:, 2]), tspan]).T
  data2 =np.concatenate((data1, P1), axis=1)
  data = np.concatenate((data2, P2), axis=1)
  return data


data = []
t = 20
dt = 0.01
th1_rand, th2_rand = np.random.uniform(-180, 180, 2)
data.append(generate_pendulum_dataset(th1_rand, th2_rand, t, dt))

# Concatenate all simulation data into a single array
all_data = np.vstack(data)

# Save to CSV
df = pd.DataFrame(all_data, columns=['theta1', 'theta2', 'time', 'P1_x', 'P1_y', 'P2_x', 'P2_y'])
csv_path = 'double_pendulum_dataset.csv'
df.to_csv(csv_path, index=False)