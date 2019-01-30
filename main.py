# importing required libraries
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import pandas
import numpy as np
import matplotlib.pyplot as plt

# for mac os
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")

# some parts coming from
# https://www.analyticsvidhya.com/blog/2018/10/predicting-stock-price-machine-learningnd-deep-learning-techniques-python/
# which is a basic lstm for stock-price prediction

# setting figure/graph/plot size
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20, 10

# sizes (4000 ticks = approximatly one round)
# first 3500 is used for training the network
# last 500 ticks are used for testing
dataSize = 4000
trainSize = 3500

# read file (needs to be in the same folder)
# currently the file contains a lot of data
# (10 players, ~4000 ticks, isAlive, name, pitch, yaw, teamNumber,x,y,z)
df_csv = pandas.read_csv("8_seadoggs-vs-breakaway-dust2.csv")


# testing sorting in data-frames
# by index
# our_data = df_csv.sort_index(ascending=True, axis=0)

# by values - we needed to sort the csv after player-names and lines
our_data = df_csv.sort_values(by=['name', 'line'], ascending=True)

# console output to test if it is sorted correctly (head(10) = print first 10 lines)
print(our_data.head(10))

# Data-frame with the same size as our original data but only two columns
# because we currently only use yaw, line for prediction
our_new_data = pandas.DataFrame(index=range(0, dataSize), columns=['line', 'yaw'])

# copy data in new data-frame (0,7 are the columns for line, yaw in the original data-frame)
# iterate over every row and fill with the value
x = 0
for i, r in our_data.head(dataSize).iterrows():
    our_new_data['line'][x] = r.values[0]
    our_new_data['yaw'][x] = r.values[7]
    x = x+1
# console-output for testing
print(our_new_data)

# setting index (line as index)
our_new_data.index = our_new_data.line
our_new_data.drop('line', axis=1, inplace=True)

# creating train and test sets
dataset = our_new_data.values
# cutting in 2 parts
# (we only use every 10th line from our data-frame)
# train = first 3500 elements (from line 0-35000)
train = our_new_data[:trainSize]
# valid = last 500 ticks (from line 35000-40000)
valid = our_new_data[trainSize:dataSize]

# plot the original data -> all yaw values for the 'dataSize' (4000) ticks
plt.figure(figsize=(16, 8))
plt.plot(train['yaw'], color='b')
plt.plot(valid['yaw'], color='b')
plt.xlabel('index (line)')
plt.ylabel('yaw')
plt.title('data from player 1 (one round)')
plt.show()

# converting dataset into x_train and y_train
# normalising (from values 0-360 -> 0-1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []

for i in range(60, len(train)):
    x_train.append(scaled_data[i-60:i, 0])
    y_train.append(scaled_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)


x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# create and fit the LSTM network
# decreasing the units leads to decreased precision (with nearly same time needed for calculation)
# we use 2 Lstm layers (and the output layer)
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

# increasing the epochs greatly increases precision (epochs=2 -> loss/2)
# more then 3 epochs doesnt really have any benefits anymore (only takes a lot more time)
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=1)
# verbose is just for visualising the progress (in console)
# change to 2 for -> number of epoch like this: epoch 1/1 ...

# predicting values, using past 60 ticks/values
inputs = our_new_data[len(our_new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)

X_test = []
for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
yaw_value = model.predict(X_test)

# scaling yaw back to 0-360 (from values between 0,1)
yaw_value = scaler.inverse_transform(yaw_value)

# mean-squared error (Quadratisches Mittel)
# value is very high because (not a good representation of our accuracy)
# we have yaw-values between 0-360
# Example:
# real value = 2/360
# predicted value = 359/360  -> difference is really high (357)
# even when the real difference (in the game) is only 3 degrees
# at the bottom of the code we tried to fix this problem and so the value is closer to the actual level of accuracy
rms = np.sqrt(np.mean(np.power((valid - yaw_value), 2)))
print('rms-value')
print(rms)


# for plotting real/predicted values
train = our_new_data[:trainSize]
valid = our_new_data[trainSize:dataSize]
valid['Predictions'] = yaw_value
plt.figure(figsize=(16, 8))
plt.plot(train['yaw'], label='train-data')
plt.plot(valid[['yaw', 'Predictions']], label='valid-data')
plt.xlabel('index (line)')
plt.ylabel('yaw')
plt.title('trained vs valid data')
plt.show()

# calculate delta

# create new d-f with one column
delta = pandas.DataFrame(index=(range(0, dataSize)), columns=['delta'])

# filter really high values
# (same problem as described above (line 128)...
# difference between 0 and 359 should be 1 ... not 359)
for index, row in valid.iterrows():
    delta['delta'][index] = abs(row['yaw'] - row['Predictions'])
    if delta['delta'][index] > 180:
        delta['delta'][index] = 360 - delta['delta'][index]


# plot delta (difference between real value and prediction)
plt.figure(figsize=(16, 8))
plt.plot(delta['delta'], color='r')
plt.xlabel('index (line)')
plt.ylabel('delta')
plt.title('delta-values (difference between real/prediction)')
plt.show()

# testing
# print values/predictions to console
print(train)
print(valid)


