import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, Lambda
from keras.losses import Huber
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

from torch.utils.data import DataLoader, Dataset
from torch.nn import Transformer
from torch import nn
import torch
import math


data = pd.read_csv(os.path.join(os.getcwd(), "webapp", "media", "377300", "377300.csv"), encoding='cp949') 

scaler = MinMaxScaler()
data['종가'] = scaler.fit_transform(data['종가'].to_numpy().reshape(-1, 1))

train = data[:-30]
train_data = train['종가'].to_numpy()

test = data[-30:]
test_data = test['종가'].to_numpy()


# from statsmodels.tsa.seasonal import seasonal_decompose
# result = seasonal_decompose(train['종가'], model="additive", period=24)
# fig = plt.figure()
# fig = result.plot()
# fig.set_size_inches(20,7)
# plt.show()


# import statsmodels.api as sm
# fig = plt.figure(figsize=(20,8))
# ax1 = fig.add_subplot(211)
# fig = sm.graphics.tsa.plot_acf(train['종가'], lags=20, ax=ax1)

# fig = plt.figure(figsize=(20,8))
# ax1 = fig.add_subplot(212)
# fig = sm.graphics.tsa.plot_pacf(train['종가'], lags=20, ax=ax1)


from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools
from tqdm import tqdm

p = range(0,3)
d = range(1,2)
q = range(0,6)
m = 15
pdq = list(itertools.product(p,d,q))
seasonal_pdq = [(x[0],x[1], x[2], m) for x in list(itertools.product(p,d,q))]

aic = []
params = []

with tqdm(total = len(pdq) * len(seasonal_pdq)) as pg:
    for i in pdq:
        for j in seasonal_pdq:
            pg.update(1)
            try:
                model = SARIMAX(train['종가'], order=(i), season_order = (j))
                model_fit = model.fit()
                aic.append(round(model_fit.aic,2))
                params.append((i,j))
            except:
                continue

optimal = [(params[i],j) for i,j in enumerate(aic) if j == min(aic)]
model_opt = SARIMAX(train['종가'], order = optimal[0][0][0], seasonal_order = optimal[0][0][1])
model_opt_fit = model_opt.fit()
model_opt_fit.summary()


model = SARIMAX(train['종가'], order=optimal[0][0][0], seasonal_order=optimal[0][0][1])
model_fit = model.fit(disp=0)
forecast = model_fit.forecast(steps=30)

plt.figure(figsize=(20,5))
plt.plot(range(0, len(data)), data['종가'].iloc[0:], label="real")
plt.plot(forecast, label="predict")
plt.legend()
plt.show()
