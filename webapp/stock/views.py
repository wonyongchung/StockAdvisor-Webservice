from django.shortcuts import render
from main.models import Main
import numpy as np
import pandas as pd
import os, glob
from sklearn.ensemble import RandomForestRegressor
from pykrx import stock as pystock
from datetime import datetime, timedelta
from pytz import timezone
from .data_update import update_data



def MinMaxScaler(data):
    denom = np.max(data,0)-np.min(data,0)            # np.min(data,0) 이었는데 - 값이 있어서 0이 더 작은 값으로 되서 0으로 나누는 경우 에러
    nume = data-np.min(data,0)
    return nume/denom

# 정규화 되돌리기 함수 
def back_MinMax(data,value):
    diff = np.max(data,0)-np.min(data,0)
    back = value * diff + np.min(data,0)
    return back 

def stock(request):
    # allstocks = []
    data_dir = os.path.join('webapp', 'media')
    time_now = datetime.now(tz=timezone('Asia/Seoul'))
    print(time_now)
    
    allstocks =  glob.glob(f'{data_dir}/*/')
    
    allstocks = [f"{pystock.get_market_ticker_name(os.path.split(os.path.split(ticker_dir)[0])[-1])}-{os.path.split(os.path.split(ticker_dir)[0])[-1]}" for ticker_dir in allstocks]
    allstocks.sort()
    return render(request, 'front/stock.html', {'allstocks': allstocks})

def stock_detail_dj(request):
    

    if request.method == 'POST':
        word = request.POST.get('stockname').split('-')[-1]
        print(word)
        predicted_stock_price = 0

        df = pd.read_csv('webapp/media/{}/{}.csv'.format(word, word), encoding='cp949')

        xy = df.iloc[:,1:5].values
        window, past, predict_day, future_price = 7, 8, 1, 5       
        trainSet, testSet = MinMaxScaler(xy[0:-past]), MinMaxScaler(xy[-past-window:])  # past 가장 최근의 과거 일주일만 보여줌

        def buildDataSet(data, window):
            xdata = []
            ydata = []
            for i in range(0, len(data) - window - predict_day):
                xdata.append(data[i:i + window])                           # 행은 10개씩, 열은 직전의 Number 값들도 같이 입력변수로 넣어준다
                ydata.append(data[i + window + predict_day - 1,:])                   # 행은 그 다음 행 하나랑, 열은 모든 feature
            return np.array(xdata), np.array(ydata)

        X_train, y_train=buildDataSet(trainSet, window)
        X_test, y_test=buildDataSet(testSet, window)

        tmp_y_train, tmp_y_test = y_train[:,-1], y_test[:,-1]   # 예측할 때는 맨 마지막 종가 예측 결과만 출력해준다

        # RF는 2차원 데이터를 입력으로 받아서 1차원으로 바꿔줘야한다.
        X_train = X_train.reshape(X_train.shape[0],7*4)
        X_test = X_test.reshape(X_test.shape[0],7*4)
        tmp_y_train = tmp_y_train.reshape(X_train.shape[0],)
        tmp_y_test = tmp_y_test.reshape(X_test.shape[0],)

        model = RandomForestRegressor()
        model.fit(X_train,y_train)

        predicted_stock_price = model.predict(X_test)[:,-1]  # 예측할 때는 맨 마지막 종가 예측 결과만 출력해준다
        print(df.iloc[-past:,-1])
        print("가자아아아ㅏ앙", df.iloc[-past:,-1])
        predicted_stock_price = back_MinMax(df.iloc[-past:,-1], predicted_stock_price)
        
        print("예측된 결과는 : ", predicted_stock_price.shape)
        print("원본 테스트 크기 :", tmp_y_test.shape)
        print("이;겆",xy.shape, model.predict(X_test)[-1].reshape(1,4).shape)
        xy = np.append(xy, model.predict(X_test)[-1].reshape(1,4), axis=0)   # 예측된 맨 마지막 결과를 넣어서 이제부터 미래 예측
        for i in range(future_price):
            trainSet, testSet = MinMaxScaler(xy[0:-past]), MinMaxScaler(xy[-past-window:])  # past 가장 최근의 과거 일주일만 가지고 새로 학습

            X_train, y_train = buildDataSet(trainSet, window)
            X_test, y_test = buildDataSet(testSet, window)
            print("테스트는?",X_test.shape )
            # RF는 2차원 데이터를 입력으로 받아서 1차원으로 바꿔줘야한다.
            X_train = X_train.reshape(X_train.shape[0],7*4)
            X_test = X_test.reshape(X_test.shape[0],7*4)
            y_train = y_train.reshape(y_train.shape[0],4)

            #model.fit(X_train, y_train)

            future_predicted_stock_price = model.predict(X_test)
            print("이거", df.iloc[-past:,])
            future_predicted_stock_price = back_MinMax(df.iloc[-past:,-1], future_predicted_stock_price)  # 여기서는 전체로 되돌려줌

            print("응", future_predicted_stock_price.shape)
            xy = np.append(xy, future_predicted_stock_price[-1].reshape(1,4), axis=0)   # 예측된 맨 마지막 결과를 넣어서 이제부터 미래 예측

        org_df1 = pd.DataFrame(back_MinMax(df.iloc[-5:,-1], tmp_y_test)).reset_index(drop=True)
        old = pd.DataFrame(predicted_stock_price)
        new = pd.DataFrame(xy[-future_price:, -1])
        df1 = pd.concat([org_df1, new], axis=0).reset_index(drop=True)
        df2 = pd.concat([old, new], axis=0).reset_index(drop=True)
        df3 = pd.DataFrame(range(0, predicted_stock_price.shape[0] + future_price, 1)).reset_index(drop=True)
        df4 = pd.concat([df3, df1, df2], axis=1)
        # df4 = df4.fillna(0)
        df = df4.values.tolist()
        print("df가 어찌되는겨", df4)
        print("더해진 5개가 있나요?", xy.shape)
        ratio = -1
        it = 0
        dt = 0
        n = len(predicted_stock_price)
        for i in range(n):
            for j in range(i + 1, n):
                if ((predicted_stock_price[j] / predicted_stock_price[i]) > ratio):
                    ratio = predicted_stock_price[j] / predicted_stock_price[i]
                    it = i
                    dt = j
        ratio=int(ratio*10000)

        showstock = pd.read_csv(os.path.join(os.getcwd(), "webapp", "media", "상장법인목록.csv"), encoding='cp949', index_col=1)
        showstock.index = [format(code, '06') for code in showstock.index]
        showstock = showstock.loc[word]
        return render(request, 'front/stock_detail_dj.html', {'showstock': showstock, 'df':df, 'ratio':ratio, 'it':it, 'dt':dt})