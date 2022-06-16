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


class windowDataset(Dataset):
    def __init__(self, y, input_window=80, output_window=20, stride=5):
        #총 데이터의 개수
        L = y.shape[0]
        #stride씩 움직일 때 생기는 총 sample의 개수
        num_samples = (L - input_window - output_window) // stride + 1

        #input과 output
        X = np.zeros([input_window, num_samples])
        Y = np.zeros([output_window, num_samples])

        for i in np.arange(num_samples):
            start_x = stride*i
            end_x = start_x + input_window
            X[:,i] = y[start_x:end_x]

            start_y = stride*i + input_window
            end_y = start_y + output_window
            Y[:,i] = y[start_y:end_y]

        X = X.reshape(X.shape[0], X.shape[1], 1).transpose((1,0,2))
        Y = Y.reshape(Y.shape[0], Y.shape[1], 1).transpose((1,0,2))
        self.x = X
        self.y = Y
        
        self.len = len(X)
    def __getitem__(self, i):
        return self.x[i], self.y[i]
    def __len__(self):
        return self.len

iw = 60 # 14
ow = 30 # 7

train_dataset = windowDataset(train_data, input_window=iw, output_window=ow, stride=1)
train_loader = DataLoader(train_dataset, batch_size=64)


class TFModel(nn.Module):
    def __init__(self,iw, ow, d_model, nhead, nlayers, dropout=0.5):
        super(TFModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers) 
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.encoder = nn.Sequential(
            nn.Linear(1, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, d_model)
        )
        
        self.linear =  nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1)
        )

        self.linear2 = nn.Sequential(
            nn.Linear(iw, (iw+ow)//2),
            nn.ReLU(),
            nn.Linear((iw+ow)//2, ow)
        ) 

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, srcmask):
        src = self.encoder(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src.transpose(0,1), srcmask).transpose(0,1)
        output = self.linear(output)[:,:,0]
        output = self.linear2(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def gen_attention_mask(x):
    mask = torch.eq(x, 0)
    return mask


device = torch.device("cuda")

lr = 1e-4
model = TFModel(30*2, 30, 512, 8, 4, 0.1).to(device) # 7*2, 7,
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


from tqdm import tqdm
epoch = 1000
model.train()
progress = tqdm(range(epoch))
for i in progress:
    batchloss = 0.0
    for (inputs, outputs) in train_loader:
        optimizer.zero_grad()
        src_mask = model.generate_square_subsequent_mask(inputs.shape[1]).to(device)
        result = model(inputs.float().to(device),  src_mask)
        loss = criterion(result, outputs[:,:,0].float().to(device))
        loss.backward()
        optimizer.step()
        batchloss += loss
    progress.set_description("loss: {:0.6f}".format(batchloss.cpu().item() / len(train_loader)))


def evaluate():
    input = torch.tensor(train_data[-30*2:]).reshape(1,-1,1).to(device).float().to(device) # -7*2
    model.eval()
    
    src_mask = model.generate_square_subsequent_mask(input.shape[1]).to(device)
    predictions = model(input, src_mask)
    return predictions.detach().cpu().numpy()


result = evaluate()
result = scaler.inverse_transform(result)[0]
real = data["종가"].to_numpy()
real = scaler.inverse_transform(real.reshape(-1,1))[:,0]


plt.plot(range(0,len(data)),real[:], label="real")
plt.plot(range(len(data)-30,len(data)),result, label="predict") # len(data)-7,len(data)
plt.legend()
plt.show()


##################################################################
##################### transformer + MA trial #####################
##################################################################

ma_data = pd.read_csv(os.path.join(os.getcwd(), "webapp", "media", "377300", "377300.csv"), encoding='cp949')

ma30 = ma_data['종가'].rolling(window=30).mean()
ma100 = ma_data['종가'].rolling(window=100).mean()

ma_data.insert(len(ma_data.columns), "MA30", ma30)
ma_data.insert(len(ma_data.columns), "MA100", ma100)




ma_data_30 = []
ma_data_100 = []
for i in range(0, 30):
    a = 0.3 * ma_data['MA30'].iloc[119+i] + 0.7 * result[i]
    b = 0.3 * ma_data['MA100'].iloc[119+i] + 0.7 * result[i]
    ma_data_30.append(a)
    ma_data_100.append(b)
    

plt.plot(ma_data.index, ma_data['종가'], label="price")
plt.plot(ma_data.index, ma_data['MA30'], label="MA30")
plt.plot(ma_data.index, ma_data['MA100'], label="MA100")
plt.legend()
plt.show()

data_30 = ma_data['날짜'][:30]

plt.plot(range(0,len(data)),real[:], label="real")
plt.plot(range(len(data)-30,len(data)),result, label="predict")
plt.plot(ma_data.index, ma_data['MA30'], label="MA30")
plt.plot(ma_data.index, ma_data['MA100'], label="MA100")
plt.plot(range(len(data)-30,len(data)), ma_data_30, label = 'tma_30')
plt.plot(range(len(data)-30,len(data)), ma_data_100, label = 'tma_100')
plt.legend()
plt.show()
