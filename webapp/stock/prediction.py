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


data = pd.read_csv(os.path.join(os.getcwd(), "webapp", "media", "377300", "377300.csv"), encoding='cp949')  ############# 각 파일을 읽어올 수 있게 변경해야함
# data = pd.read_csv("377300.csv")

# 날짜와 종가만 추출
# data = data.drop(columns = ["시가", "고가", "저가", "거래량"])
# print(data)

# data = data.rename(columns = {'날짜' : 'days', '종가' : 'price'})

sc = MinMaxScaler()
sc_columns = ['시가', '고가', '저가', '종가', '거래량']
data_scaled = sc.fit_transform(data[sc_columns])
data = pd.DataFrame(data_scaled, columns = sc_columns)

# data['종가'] = sc.fit_transform(data['종가'].to_numpy().reshape(-1, 1))

x_train, x_test, y_train, y_test = train_test_split(data.drop('종가', 1), data['종가'], test_size=0.2, random_state=0, shuffle=False)

def windowed_dataset(series, window_size, batch_size, shuffle):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.map(lambda w: (w[:-1], w[-1]))
    return ds.batch(batch_size).prefetch(1)

WINDOW_SIZE=7
BATCH_SIZE=32

train_data = windowed_dataset(y_train, WINDOW_SIZE, BATCH_SIZE, True)
test_data = windowed_dataset(y_test, WINDOW_SIZE, BATCH_SIZE, False)

model = Sequential([
    # 1차원 feature map 생성
    Conv1D(filters=32, kernel_size=5,
           padding="causal",
           activation="relu",
           input_shape=[WINDOW_SIZE, 1]),
    # LSTM
    LSTM(16, activation='tanh'),
    Dense(16, activation="relu"),
    Dense(1),
])

loss = Huber()
optimizer = Adam(0.0005)
model.compile(loss=Huber(), optimizer=optimizer, metrics=['mse'])

# earlystopping은 10번 epoch통안 val_loss 개선이 없다면 학습을 멈춥니다.
earlystopping = EarlyStopping(monitor='val_loss', patience=15)
# val_loss 기준 체크포인터도 생성합니다.
filename = os.path.join('tmp', 'ckeckpointer.ckpt')  ############################################ 여기에 각 media 파일에 넣어주는 작업이 필요할듯 함
checkpoint = ModelCheckpoint(filename, 
                             save_weights_only=True, 
                             save_best_only=True, 
                             monitor='val_loss', 
                             verbose=1)

history = model.fit(train_data, 
                    validation_data=(test_data), 
                    epochs=50, 
                    callbacks=[checkpoint, earlystopping])

model.load_weights(filename)
pred = model.predict(test_data)

plt.figure(figsize=(12, 9))
plt.plot(np.asarray(y_test)[7:], label='actual')
plt.plot(pred, label='prediction')
plt.legend()
plt.show()
