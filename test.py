# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 18:49:55 2023

@author: Soumyajit Saha
"""

# import numpy as np
# import matplotlib.pyplot as plt
# from statsmodels.tsa.arima.model import ARIMA

# # Generate Lorenz 96 data
# def generate_lorenz96(N, F):
#     """Generate N steps of Lorenz 96 with forcing F."""
#     x = np.zeros(N)
#     y = np.zeros(N)
#     z = np.zeros(N)
#     x[0], y[0], z[0] = (np.random.randn(3)*0.1)
#     for i in range(1, N):
#         xdot = (-x[i-2] * y[i-1]) + (x[i-1] * y[i-2]) - x[i-1] + F
#         ydot = (x[i-2] * x[i-1]) - (y[i-2] * y[i-1])
#         zdot = (x[i-2] * y[i-1]) - (z[i-1] * y[i-1])
#         x[i] = x[i-1] + (xdot * 0.01)
#         y[i] = y[i-1] + (ydot * 0.01)
#         z[i] = z[i-1] + (zdot * 0.01)
#     return x

# # Generate Lorenz 96 data with F=8
# N = 1000
# F = 8.0
# data = generate_lorenz96(N, F)

# # Split data into training and test sets
# train_size = int(len(data) * 0.8)
# train_data, test_data = data[:train_size], data[train_size:]

# # Fit ARIMA model to training data
# model = ARIMA(train_data, order=(3, 1, 0))
# model_fit = model.fit()

# # Make predictions on test data
# predictions = model_fit.forecast(steps=len(test_data))[0]

# # Plot predictions and test data
# plt.plot(test_data, label='Actual')
# plt.plot(predictions, label='Predicted')
# plt.legend()
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from statsmodels.tsa.api import VAR

# # Generate multivariate Lorenz 96 data
# def generate_multivariate_lorenz96(N, F, num_vars):
#     """Generate N steps of multivariate Lorenz 96 with forcing F and num_vars variables."""
#     data = np.zeros((N, num_vars))
#     for j in range(num_vars):
#         x = np.zeros(N)
#         y = np.zeros(N)
#         z = np.zeros(N)
#         x[0], y[0], z[0] = (np.random.randn(3)*0.1)
#         for i in range(1, N):
#             xdot = (-x[i-2] * y[i-1]) + (x[i-1] * y[i-2]) - x[i-1] + F
#             ydot = (x[i-2] * x[i-1]) - (y[i-2] * y[i-1])
#             zdot = (x[i-2] * y[i-1]) - (z[i-1] * y[i-1])
#             x[i] = x[i-1] + (xdot * 0.01)
#             y[i] = y[i-1] + (ydot * 0.01)
#             z[i] = z[i-1] + (zdot * 0.01)
#         data[:,j] = x
#     return data

# # Generate multivariate Lorenz 96 data with F=8 and num_vars=2
# N = 1000
# F = 8.0
# num_vars = 2
# data = generate_multivariate_lorenz96(N, F, num_vars)

# # Split data into training and test sets
# train_size = int(len(data) * 0.8)
# train_data, test_data = data[:train_size], data[train_size:]

# # Fit VARIMA model to training data
# model = VAR(train_data)
# model_fit = model.fit(2)

# # Make predictions on test data
# predictions = model_fit.forecast(train_data[-2:], steps=len(test_data))

# # Plot predictions and test data
# for j in range(num_vars):
#     plt.plot(test_data[:,j], label='Actual ' + str(j+1))
#     plt.plot(predictions[:,j], label='Predicted ' + str(j+1))
# plt.legend()
# plt.show()

# import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense, Conv1D, Flatten

# # Generate time series data
# T = 1000
# time = np.arange(T)
# x = np.sin(0.05 * time) + np.random.normal(0, 0.1, size=T)

# # Prepare input and output data
# window_size = 10
# input_data = np.zeros((T - window_size, window_size))
# output_data = np.zeros((T - window_size, 1))

# for i in range(T - window_size):
#     input_data[i] = x[i:i + window_size]
#     output_data[i] = x[i + window_size]

# input_data = input_data.reshape(input_data.shape[0], input_data.shape[1], 1)
# output_data = output_data.reshape(output_data.shape[0], output_data.shape[1])

# # Define and train CNN model
# model = Sequential()
# model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(window_size, 1)))
# model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
# model.add(Flatten())
# model.add(Dense(1))
# model.compile(loss='mse', optimizer='adam')
# model.fit(input_data, output_data, epochs=50, batch_size=32)

# # Generate predictions
# future_time = np.arange(T, T + 100)
# future_input_data = np.zeros((100, window_size))
# future_input_data[0] = x[-window_size:]
# future_input_data = future_input_data.reshape(future_input_data.shape[0], future_input_data.shape[1], 1)

# future_predictions = []

# for i in range(100):
#     prediction = model.predict(future_input_data)[0, 0]
#     future_predictions.append(prediction)
#     future_input_data = np.roll(future_input_data, -1, axis=1)
#     future_input_data[0, -1] = prediction

# # Plot original data and predictions
# import matplotlib.pyplot as plt

# plt.plot(time, x, label='Original Data')
# plt.plot(future_time, future_predictions, label='Predictions')
# plt.legend()
# plt.show()

# import numpy as np
# from keras.models import Sequential
# from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# # Generate time series data of Lorenz96 model
# def lorenz96(X, F):
#     return np.roll(X, -1) * (X - np.roll(X, 1)) - X + F

# def generate_data(N, F):
#     X = np.zeros((N, 20))
#     X[:, 9] = 1.0
#     for i in range(1, N):
#         X[i, :] = lorenz96(X[i - 1, :], F)
#     return X

# data = generate_data(1000, 8.0)

# # Preprocess the data
# lookback = 10
# input_data = []
# output_data = []
# for i in range(lookback, len(data)):
#     input_data.append(data[i - lookback:i, :])
#     output_data.append(data[i, :])
# input_data = np.array(input_data)
# output_data = np.array(output_data)
# input_data_mean = np.mean(input_data, axis=0)
# input_data_std = np.std(input_data, axis=0)
# input_data = (input_data - input_data_mean) / input_data_std

# # Split the preprocessed data into training and validation sets
# train_ratio = 0.7
# train_size = int(train_ratio * input_data.shape[0])
# train_input_data = input_data[:train_size, :, :]
# train_output_data = output_data[:train_size, :]
# val_input_data = input_data[train_size:, :, :]
# val_output_data = output_data[train_size:, :]

# # Define a CNN model architecture
# model = Sequential()
# model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(lookback, 20)))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Flatten())
# model.add(Dense(50, activation='relu'))
# model.add(Dense(20))

# # Train the model on the training set
# model.compile(optimizer='adam', loss='mse')
# model.fit(train_input_data, train_output_data, epochs=50, batch_size=32, validation_data=(val_input_data, val_output_data))

# # Evaluate the model on the validation set
# score = model.evaluate(val_input_data, val_output_data, batch_size=32)
# print('Validation loss:', score)

# # Use the trained model to predict future points
# future_data = generate_data(100, 8.0)
# future_input_data = (future_data[-lookback:, :] - input_data_mean) / input_data_std
# future_output_data = model.predict(np.array([future_input_data]))

# import matplotlib.pyplot as plt

# # Plot the predicted future data
# future_data = np.concatenate((future_data, future_output_data), axis=0)
# plt.plot(future_data[:, 0], label='Variable 1')
# # plt.plot(future_data[:, 1], label='Variable 2')
# # plt.plot(future_data[:, 2], label='Variable 3')
# plt.legend()
# plt.show()


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense

# Define Lorenz96 model
def lorenz96(x, F):
    dx = np.zeros_like(x)
    dx[0] = (x[1] - x[-2]) * x[-1] - x[0] + F
    dx[1] = (x[2] - x[-1]) * x[0] - x[1] + F
    dx[-1] = (x[0] - x[-3]) * x[-2] - x[-1] + F
    dx[2:-1] = (x[3:] - x[:-3]) * x[1:-2] - x[2:-1] + F
    return dx

# Generate data
np.random.seed(0)
t = 3000
F = 8
N = 40
x0 = np.random.randn(N)
x = np.zeros((t, N))
x[0] = x0
for i in range(1, t):
    x[i] = x[i-1] + lorenz96(x[i-1], F) * 0.01

# Split data into input/output sequences
n_steps = 10
X = np.zeros((t-n_steps, n_steps, N, 1))
y = np.zeros((t-n_steps, N))
for i in range(n_steps, t):
    X[i-n_steps] = x[i-n_steps:i, :, np.newaxis]
    y[i-n_steps] = x[i]

# Split data into training/validation sets
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_val, y_val = X[split:], y[split:]


# Define CNN model
model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(n_steps, N, 1)),
    Flatten(),
    Dense(N)
])

# Compile model
model.compile(loss='mse', optimizer='adam')

# Train model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50)



# Evaluate predictions
y_pred = model.predict(X_val)
mse = np.mean((y_val - y_pred)**2)
print("Validation MSE:", mse)

# Plot predictions
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(y_val[:, 0], label="True")
plt.plot(y_pred[:, 0], label="Predicted")
plt.legend()
plt.show()
