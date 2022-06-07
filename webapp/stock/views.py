from django.shortcuts import render
from .models import Stock
from main.models import Main
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, glob
from tensorflow.keras.models import model_from_json
from sklearn.ensemble import RandomForestRegressor
from pykrx import stock as pystock

def MinMaxScaler(data):
    denom = np.max(data,0)-np.min(data,0)            # np.min(data,0) 이었는데 - 값이 있어서 0이 더 작은 값으로 되서 0으로 나누는 경우 에러
    nume = data-np.min(data,0)
    return nume/denom

def stock(request):
    site = Main.objects.get(pk=2)
    # allstocks = Stock.objects.all()         #주식 데이터베이스 전부 -> csv로 변경 필요
    # print(os.getcwd())
    # allstocks = pd.read_csv(os.path.join(os.getcwd(), "webapp", "media", "상장법인목록.csv"), encoding='cp949', index_col=0)
    #media 폴더 내에 저장되어 있는 회사폴더명 불러와서 정렬 후 list로 반환
    allstocks = []
    rootdir = os.path.join('webapp', 'media')
    allstocks =  glob.glob(f'{rootdir}/*/')
    
    allstocks = [f"{pystock.get_market_ticker_name(os.path.split(os.path.split(ticker_dir)[0])[-1])}-{os.path.split(os.path.split(ticker_dir)[0])[-1]}" for ticker_dir in allstocks]
    allstocks.sort()
    return render(request, 'front/stock.html', {'site': site, 'allstocks': allstocks})

def stock_detail_dj(request):
    if request.method == 'POST':
        word = request.POST.get('stockname').split('-')[-1]
        print(word)
        predicted_stock_price = 0

        df = pd.read_csv('webapp/media/{}/{}.csv'.format(word, word), encoding='cp949')

        xy = df.iloc[:,1:5].values       
        window = 7                     
        trainSize = int(len(df)*0.8)  
        print("훈련 사이즈는~?", trainSize)                  
        print("첫번째 행 잘 뽑히나 보자", xy[0])
        trainSet = MinMaxScaler(xy[0:trainSize])
        testSet = MinMaxScaler(xy[trainSize-window:])             # trainsize - windowsize 부터 끝까지 test

        predict_day = 1

        def buildDataSet(data, window):
            xdata = []
            ydata = []
            for i in range(0, len(data) - window - predict_day):
                xdata.append(data[i:i + window])                           # 행은 10개씩, 열은 직전의 Number 값들도 같이 입력변수로 넣어준다
                ydata.append(data[i + window + predict_day - 1,[-1]])                   # 행은 그 다음 행 하나랑, 열은 Number만
            return np.array(xdata), np.array(ydata)

        X_train, y_train=buildDataSet(trainSet, window)
        X_test, y_test=buildDataSet(testSet, window)

        # RF는 2차원 데이터를 입력으로 받아서 1차원으로 바꿔줘야한다.
        X_train = X_train.reshape(X_train.shape[0],7*4)
        X_test = X_test.reshape(X_test.shape[0],7*4)
        y_train = y_train.reshape(X_train.shape[0],)
        y_test = y_test.reshape(X_test.shape[0],)

        model = RandomForestRegressor()
        print("훈련 : ", X_train.shape, y_train.shape)
        model.fit(X_train,y_train)

        predicted_stock_price = model.predict(X_test)
        print("예측된 결과는 : ", predicted_stock_price.shape)
        print("원본 테스트 크기 :", y_test.shape)

        df1 = pd.DataFrame(y_test)
        df2 = pd.DataFrame(predicted_stock_price)
        df3 = pd.DataFrame(range(0, predicted_stock_price.shape[0], 1))
        df4 = pd.concat([df3, df1, df2], axis=1)
        df = df4.values.tolist()

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
        # .loc[word]
        # showstock.loc['종목코드'] = format(showstock['종목코드'].copy(), '06')
        print(showstock)
        return render(request, 'front/stock_detail_dj.html', {'showstock': showstock, 'df':df, 'ratio':ratio, 'it':it, 'dt':dt})