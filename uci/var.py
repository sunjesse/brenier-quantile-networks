from statsmodels.tsa.api import VAR
import numpy as np
from dataloader import MinMaxScaler
from utils import *

#data = np.loadtxt('./data/stock/stock_data.csv', delimiter=',', skiprows=1)
#data = data[::-1]
data = np.load('./data/energy/data_24.npy')
#data = MinMaxScaler(data)
n = data.shape[0]
train, test = data[:7*n//10], data[4*n//5:]
train = np.concatenate(train, axis=0)
print(train.shape)

model = VAR(train)

results = model.fit()

#y_hat = results.forecast(test, )
y = []
pred = []
for i in range(test.shape[0]):#len(test)-28):
    y_hat = results.forecast(test[i][:-1], 1)#i: i+23], 5)
    y.append([test[i][-1]])#+23: i+28])
    pred.append(y_hat)

print(len(y), len(pred))
y = np.concatenate(y, axis=0)
pred = np.concatenate(pred, axis=0)
print(y, pred)
print(y.shape, pred.shape)
err = np.abs(y - pred)
print(y.shape, pred.shape)
print('rmse: ' + str(rmse(y, pred)))
print('smape: ' + str(smape(y, pred)))
print('max ae: ' + str(err.max()))
print('mean ae: ' + str(err.mean()))
