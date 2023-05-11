# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 10:52:24 2023

@author: Soumyajit Saha
"""

from L96 import *
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from sklearn.metrics import mean_squared_error
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense

# Load initial conditions for L96 model
initX, initY = np.load('./initX.npy'), np.load('./initY.npy')
np.random.seed(123)

N = 36

l96_two = L96TwoLevel(K = N, save_dt=0.001, X_init=initX, Y_init=initY, noYhist=True)

l96_two.iterate(10)

h2 = l96_two.history


t = len(h2.X)

n_steps = 10
X = np.zeros((t-n_steps, n_steps, N, 1))
y = np.zeros((t-n_steps, N))
for i in range(n_steps, t):
    X[i-n_steps] = np.array(h2.B)[i-n_steps:i, :, np.newaxis]
    y[i-n_steps] = np.array(h2.B)[i]

# Split data into training/validation sets
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_val, y_val = X[split:], y[split:]


model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(n_steps, N, 1)),
    Flatten(),
    Dense(N)
])

model.compile(loss='mse', optimizer='adam')

model.fit(X_train, y_train, epochs=50)



y_pred = model.predict(X_val)
mse = np.mean((y_val - y_pred)**2)
print("Validation MSE:", mse)

avg_rmse = 0

for j in range(N):
    plt.figure(figsize=(10, 5))
    plt.plot(y_val[:, j], label="Actual " + str(j+1), color='green')
    plt.plot(y_pred[:, j], label="Predicted " + str(j+1), color='red')
    plt.xlabel("Time")
    plt.ylabel("B")
    plt.legend()
    plt.title('CNN PREDICTIONS')
    plt.show()
    print('RMSE for ' + str(j) + ' variable: ' + str(mean_squared_error(y_val[:, j], y_pred[:, j])))
    avg_rmse = mean_squared_error(y_val[:, j], y_pred[:, j])
    
avg_rmse = avg_rmse / N
print(avg_rmse)
    
    
X = np.zeros((t-n_steps, n_steps, N, 1))
y = np.zeros((t-n_steps, N))
for i in range(n_steps, t):
    X[i-n_steps] = np.array(h2.Y_mean)[i-n_steps:i, :, np.newaxis]
    y[i-n_steps] = np.array(h2.Y_mean)[i]

# Split data into training/validation sets
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_val, y_val = X[split:], y[split:]


model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(n_steps, N, 1)),
    Flatten(),
    Dense(N)
])

model.compile(loss='mse', optimizer='adam')

model.fit(X_train, y_train, epochs=50)



y_pred = model.predict(X_val)
mse = np.mean((y_val - y_pred)**2)
print("Validation MSE:", mse)

avg_rmse = 0

for j in range(N):
    plt.figure(figsize=(10, 5))
    plt.plot(y_val[:, j], label="Actual " + str(j+1), color='green')
    plt.plot(y_pred[:, j], label="Predicted " + str(j+1), color='red')
    plt.xlabel("Time")
    plt.ylabel("Y_mean")
    plt.legend()
    plt.title('CNN PREDICTIONS')
    plt.show()
    print('RMSE for ' + str(j) + ' variable: ' + str(mean_squared_error(y_val[:, j], y_pred[:, j])))
    avg_rmse = mean_squared_error(y_val[:, j], y_pred[:, j])
    
avg_rmse = avg_rmse / N
print(avg_rmse)