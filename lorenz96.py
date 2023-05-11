# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 20:54:12 2023

@author: Soumyajit Saha
"""

from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense




# These are our constants
N = 50  # Number of variables
F = 8  # Forcing



def L96(x, t):
    """Lorenz 96 model with constant forcing"""
    # Setting up vector
    d = np.zeros(N)
    # Loops over indices (with operations and Python underflow indexing handling edge cases)
    for i in range(N):
        d[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
    return d


x0 = F * np.ones(N)  # Initial state (equilibrium)
x0[0] += 0.01  # Add small perturbation to the first variable
t = np.arange(0.0, 300.0, 0.01)

x = odeint(L96, x0, t)

# Plot the first three variables
fig = plt.figure()
ax = fig.gca(projection="3d")
ax.plot(x[:, 0], x[:, 1], x[:, 2])
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$x_3$")
plt.show()

# model = LinearRegression()
# model.fit(t[: int(0.8 * len(t))].reshape(-1,1), x[: int(0.8 * len(t))])

# t_test = np.array([[2401]])

# x_pred = model.predict(t_test)

x_train = x[: int(0.8 * len(t))]
x_test = x[int(0.8 * len(t)) :]

# model = ARIMA(x_train, order=(3, 1, 0))
# model_fit = model.fit()

# predictions = model_fit.forecast(steps=len(x_test))[0]

# plt.plot(x_test, label='Actual')
# plt.plot(predictions, label='Predicted')
# plt.legend()
# plt.show()

model = VAR(x_train)
model_fit = model.fit(2)

predictions = model_fit.forecast(x_train[-2:], steps=len(x_test))
avg_rmse = 0

for j in range(N):
    plt.figure(figsize=(10, 5))
    plt.plot(x_test[:,j], label='Actual ' + str(j+1), color='green')
    plt.plot(predictions[:,j], label='Predicted ' + str(j+1), color='red')
    plt.title('ARIMA PREDICTIONS')
    plt.legend()
    plt.show()
    print('RMSE for ' + str(j) + ' variable: ' + str(mean_squared_error(x_test[:, j], predictions[:, j])))
    avg_rmse += mean_squared_error(x_test[:, j], predictions[:, j])
    
avg_rmse = avg_rmse / N
print(avg_rmse)
    
    
# for j in range(N):
#     plt.plot(x_train[:,j], label='Actual ' + str(j+1))
#     plt.legend()
#     plt.show()


# USING CNN

n_steps = 10
X = np.zeros((t.size-n_steps, n_steps, N, 1))
y = np.zeros((t.size-n_steps, N))
for i in range(n_steps, t.size):
    X[i-n_steps] = x[i-n_steps:i, :, np.newaxis]
    y[i-n_steps] = x[i]

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
    plt.legend()
    plt.title('CNN PREDICTIONS')
    plt.show()
    print('RMSE for ' + str(j) + ' variable: ' + str(mean_squared_error(y_val[:, j], y_pred[:, j])))
    avg_rmse += mean_squared_error(y_val[:, j], y_pred[:, j])

avg_rmse = avg_rmse / N
print(avg_rmse)
    