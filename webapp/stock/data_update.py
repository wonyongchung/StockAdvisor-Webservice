import pandas as pd
from datetime import datetime, timedelta
from pytz import timezone
from pykrx import stock
import glob, os
from celery import shared_task
import numpy as np
#from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import math
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, Lambda
from keras.losses import Huber
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from django.shortcuts import render
import numpy as np
import pandas as pd
import os, glob
from sklearn.ensemble import RandomForestRegressor
from pykrx import stock as pystock

def MinMaxScaler(data):
    denom = np.max(data,0)-np.min(data,0)            # np.min(data,0) 이었는데 - 값이 있어서 0이 더 작은 값으로 되서 0으로 나누는 경우 에러
    nume = data-np.min(data,0)
    return nume/denom

# 정규화 되돌리기 함수 
def back_MinMax(data,value):
    diff = np.max(data,0)-np.min(data,0)
    back = value * diff + np.min(data,0)
    return back 

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

    total = []

    for media in market_cap:
        media_dir = os.path.join(data_dir , f"{media}")
        df = pd.read_csv(f"{media_dir}/{media}.csv", encoding='cp949')

        xy = df.iloc[:,-2].values   # 종가만 가져다가 예측

        window, future_price  = 90, 30  
        train_size = int(len(xy)) - window
        trainSet = MinMaxScaler(xy) # past 가장 최근의 과거 일주일만 보여줌

        def buildDataSet(data, window):
            xdata, ydata = [], []
            for i in range(0, len(data) - window):
                xdata.append(data[i:i + window])                           # 행은 10개씩, 열은 직전의 Number 값들도 같이 입력변수로 넣어준다
                ydata.append(data[i + window])                   # 행은 그 다음 행 하나랑, 열은 모든 feature
            return np.array(xdata), np.array(ydata)

        X_train, y_train=buildDataSet(trainSet, window)

        # RF는 2차원 데이터를 입력으로 받아서 1차원으로 바꿔줘야한다.
        X_train = X_train.reshape(X_train.shape[0],window)

        print("맨 처음 훈련 셋 좀 보자", X_train.shape,  y_train.shape)
        model = RandomForestRegressor()
        model.fit(X_train,y_train)   # 예측할 때는 맨 마지막 종가 예측 결과만 출력해준다

        predicted_stock_price = model.predict(X_train[-1:])  # 예측할 때는 맨 마지막 종가 예측 결과만 출력해준다
        predicted_stock_price = back_MinMax(df.iloc[train_size-window:,-2], predicted_stock_price)
        print("맨 처음에 예측한 미래의 결과는?", predicted_stock_price.shape)
        xy = np.append(xy, predicted_stock_price)   # 예측된 맨 마지막 결과를 넣어서 이제부터 미래 예측
        for p in range(future_price):
            print("현재 총 데이터셋은 ", xy.shape)   # 원래 1228개 있다가 1229개로 늘어남
            trainSet = MinMaxScaler(xy)   # past 가장 최근의 과거 window 사이즈만 가지고 새로 학습
            X_train, y_train = buildDataSet(trainSet, window)

            # RF는 2차원 데이터를 입력으로 받아서 1차원으로 바꿔줘야한다.
            X_train = X_train.reshape(X_train.shape[0],window)
            y_train = y_train.reshape(y_train.shape[0],)

            future_predicted_stock_price = model.predict(X_train[-1:])
            future_predicted_stock_price = back_MinMax(df.iloc[train_size-window:,-2], future_predicted_stock_price)  # 여기서는 전체로 되돌려줌
            print("추가할 정보", future_predicted_stock_price.shape)
            xy = np.append(xy, future_predicted_stock_price, axis=0)   # 예측된 맨 마지막 결과를 넣어서 이제부터 미래 예측

        total.append(xy[-future_price:])

    total = pd.DataFrame(total).transpose()
    #total = total.astype(int)
    total.to_csv(f"{data_dir}/predict.csv", encoding='cp949', header=False, index=False)

if __name__=="__main__":
    update_data()