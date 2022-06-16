import pandas as pd
from datetime import datetime, timedelta
from pytz import timezone
from pykrx import stock
import glob, os
from celery import shared_task

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.nn import Transformer
from torch import nn
import torch
import math
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, Lambda
from keras.losses import Huber
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

@shared_task
def update_data():
    
    time_now = datetime.now(tz=timezone('Asia/Seoul'))
    time_now_str = time_now.strftime(format='%Y%m%d')
    # ticker_list = stock.get_market_ticker_list(date=time_now, market='KOSPI')
    time_start = time_now - timedelta(days=365*5)       #5년전부터 가져오기
    time_start_str = time_start.strftime(format='%Y%m%d')

    # data_dir = os.path.join("webapp", 'media')
    data_dir = os.path.join('media')

    print(f"{time_now_str} - Updating Stock Info...")

    # 전체 주식 시가총액 불러오기
    # market_cap = stock.get_market_cap(date=time_now_str, prev=True,  market='KOSPI')  #시가총액 순으로 정렬된 dataframe 반환
    # market_cap['종목이름'] = [stock.get_market_ticker_name(tk) for tk in market_cap.index]

    # 현재 가지고 있는 기업들만 업데이트
    # market_cap =  glob.glob(f'{data_dir}/*/')
    market_cap =  ['000270','000660','000810','003490','003550','003670','005380',
                '005490','005930','005935','006400','009150','009830','010130',
                '010950','011070','011200','012330','015760','017670','018260',
                '024110','028260','030200','032830','033780','034020','034730',
                '035420','035720','036570','051900','051910','055550','066570',
                '068270','086790','090430','096770','105560','207940','259960',
                '302440','316140','323410','329180','352820','361610','373220','377300']
    
    # market_cap = market_cap[:50]
    # ticker로 5년간 데이터 불러오기

    for ticker in market_cap:
        ticker_dir = os.path.join(data_dir , f"{ticker}")
        # print(ticker)
        
        try:                                        #회사 폴더 없으면 생성
            os.mkdir(ticker_dir)       
        except:                                     #있으면 말고
            pass

        data = stock.get_market_ohlcv_by_date(fromdate=time_start_str, todate=time_now_str, ticker=ticker)
        data.to_csv(f"{ticker_dir}/{ticker}.csv", encoding='cp949')


    for media in market_cap:
        media_dir = os.path.join(data_dir , f"{media}")
        data = pd.read_csv(f"{media_dir}/{media}.csv", encoding='cp949')
        
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
    
        ma_data = pd.read_csv('./{}/{}'.format(media, media), encoding='cp949')
        ma30 = ma_data['종가'].rolling(window=100).mean()
        ma_data.insert(len(ma_data.columns), "MA100", ma30)
        ma_data_30 = []
        for i in range(0, 30):
            a = 0.3 * ma_data['MA30'].iloc[119+i] + 0.7 * result[i]
            ma_data_30.append(a)

        ma_data_30.to_csv(f"{media_dir}/predict.csv", encoding='cp949')

if __name__=="__main__":
    update_data()
