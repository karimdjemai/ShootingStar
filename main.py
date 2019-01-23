# importing required libraries

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import pandas

from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt
# some parts coming from
# https://www.analyticsvidhya.com/blog/2018/10/predicting-stock-price-machine-learningnd-deep-learning-techniques-python/
# which is a basic lstm for stock-price prediction

# setting figure size
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20, 10

# for normalizing data
#scaler = MinMaxScaler(feature_range=(0, 1))

# read file (needs to be in the same folder)
# df = pandas.read_csv('NSE-BSE.csv')
df_csv = pandas.read_csv("8_seadoggs-vs-breakaway-dust2.csv")
# df_csv = pandas.read_csv("8_seadoggs-vs-breakaway-dust2.csv", index_col=[2, 0])

# print(df.head())

# setting index as ____ (date, line, ...)
# df['Date'] = pandas.to_datetime(df.Date, format='%Y-%m-%d')
# df.index = df['Date']
# df_csv.index = df_csv['line']

# plot
# plt.figure(figsize=(16, 8))
# plt.plot(df['Close'], label='Close Price history')
# plt.xlabel('index (line)')
# plt.ylabel('yaw')
# plt.title('trying stuff')
# plt.plot(df_csv['line'], df_csv['yaw'], label='yaw history')
# plt.show()


# creating dataframe with date and the target variable
# data = df.sort_index(ascending=True, axis=0)
# our_data = df_csv.sort_index(ascending=True, axis=0)

# df_csv.index = df_csv['name'] -> moved to import ... ^
#our_data = df_csv.sort_index(ascending=True, axis=0)


our_data = df_csv.sort_values(by=['name', 'line'], ascending=True)
print(our_data.head(10))

# new_data = pandas.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])
our_new_data = pandas.DataFrame(index=range(0, 4000), columns=['line', 'yaw'])

'''
for i in range(0, len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]
 

# for i in range(0, len(our_data)):
for i in range(0, 400):
    our_new_data['line'][i] = our_data['line'][i]
    our_new_data['yaw'][i] = our_data['yaw'][i]
'''


x = 0

for i, r in our_data.head(4000).iterrows():
    our_new_data['line'][x] = r.values[0]
    our_new_data['yaw'][x] = r.values[7]
    x = x+1
print(our_new_data)



# setting index
# new_data.index = our_new_data.Date
our_new_data.index = our_new_data.line
our_new_data.drop('line', axis=1, inplace=True)

# print(new_data)

# creating train and test sets
dataset = our_new_data.values

#train = dataset[0:3500, :]
#valid = dataset[3500:4000, :]

# print
train = our_new_data[:3500]
valid = our_new_data[3500:4000]

plt.figure(figsize=(16, 8))
plt.plot(train['yaw'])
plt.plot(valid['yaw'])
plt.show()

# converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []

for i in range(60, len(train)):
    x_train.append(scaled_data[i-60:i, 0])
    y_train.append(scaled_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

#Todo: verstehen
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

#todo: verstehen
# predicting 246 values, using past 60 from the train data
inputs = our_new_data[len(our_new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)


X_test = []
for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
yaw_value = model.predict(X_test)
yaw_value = scaler.inverse_transform(yaw_value)

# Results
rms = np.sqrt(np.mean(np.power((valid - yaw_value), 2)))
print(rms)


# for plotting
train = our_new_data[:3500]
valid = our_new_data[3500:4000]
valid['Predictions'] = yaw_value
plt.figure(figsize=(16, 8))
plt.plot(train['yaw'])
plt.plot(valid[['yaw', 'Predictions']])
plt.show()


# plot delta
delta = pandas.DataFrame(index=(range(0,4000)), columns=['delta'])


for index, row in valid.iterrows():
    delta['delta'][index] = abs(row['yaw'] - row['Predictions'])


plt.figure(figsize=(16,8))
plt.plot(delta['delta'])
plt.show()


# print values/predictions to console
#print(train.head(10))
#print(valid)


