from django.shortcuts import render
from main.models import Main
import numpy as np
import pandas as pd
import os, glob
from pykrx import stock as pystock
from datetime import datetime, timedelta
from pytz import timezone
import random
from sklearn.ensemble import RandomForestRegressor

def home(request):
    rand = random.random()
    return render(request, 'front/home.html', {'rand':rand})

def stock(request):
    # allstocks = []
    data_dir = os.path.join('webapp', 'media')
    time_now = datetime.now(tz=timezone('Asia/Seoul'))
    print(time_now)
    
    allstocks =  glob.glob(f'{data_dir}/*/')
    
    allstocks = [f"{pystock.get_market_ticker_name(os.path.split(os.path.split(ticker_dir)[0])[-1])}-{os.path.split(os.path.split(ticker_dir)[0])[-1]}" for ticker_dir in allstocks]
    #allstocks.sort()
    monthafter = "webapp/media/predict.csv"
    monthafterdata = pd.read_csv(monthafter).iloc[29,:]
    # print(monthafterdata[0])
    monthrate = []
    nowlist = []
    route = "webapp/media/"
    for stock in allstocks:
        print(stock)
        namelist = stock.split("-")
        num = namelist[-1]
        name = "".join(namelist[:-1])
        temproute = route+num+"/"+num+".csv"
        temp = pd.read_csv(temproute, encoding='cp949')
        nowlist.append([temp["종가"][len(temp)-1], name, num])
        
    for idx,now in enumerate(nowlist):
        monthrate.append([round(100*(monthafterdata[idx]-now[0])/now[0],1), now[1], now[0]])
    monthrate.sort()
    lowfive = monthrate[:5]
    highfive = monthrate[-5:]
    
    return render(request, 'front/stock.html', {'allstocks': allstocks,
                                                'h1name': highfive[-1][1],
                                               'h1rate':highfive[-1][0],
                                                'h1price':highfive[-1][2],
                                                'h2name': highfive[-2][1],
                                               'h2rate':highfive[-2][0],
                                                'h2price':highfive[-2][2],
                                                'h3name': highfive[-3][1],
                                               'h3rate':highfive[-3][0],
                                                'h3price':highfive[-3][2],
                                                'h4name': highfive[-4][1],
                                               'h4rate':highfive[-4][0],
                                                'h4price':highfive[-4][2],
                                                'h5name': highfive[-5][1],
                                               'h5rate':highfive[-5][0],
                                                'h5price':highfive[-5][2],
                                                'l1name': lowfive[0][1],
                                               'l1rate':lowfive[0][0],
                                                'l1price':lowfive[0][2],
                                                'l2name': lowfive[1][1],
                                               'l2rate':lowfive[1][0],
                                                'l2price':lowfive[1][2],
                                                'l3name': lowfive[2][1],
                                               'l3rate':lowfive[2][0],
                                                'l3price':lowfive[2][2],
                                                'l4name': lowfive[3][1],
                                               'l4rate':lowfive[3][0],
                                                'l4price':lowfive[3][2],
                                                'l5name': lowfive[4][1],
                                               'l5rate':lowfive[4][0],
                                                'l5price':lowfive[4][2]})

def MinMaxScaler(data):
    denom = np.max(data,0)-np.min(data,0)            # np.min(data,0) 이었는데 - 값이 있어서 0이 더 작은 값으로 되서 0으로 나누는 경우 에러
    nume = data-np.min(data,0)
    return nume/denom

# 정규화 되돌리기 함수 
def back_MinMax(data,value):
    diff = np.max(data,0)-np.min(data,0)
    back = value * diff + np.min(data,0)
    return back 

def stock_detail_dj(request):
    if request.method == 'POST':
        word = request.POST.get('stockname').split('-')[-1]
        quantity = request.POST.get('quantity')
        print(quantity)
        df = pd.read_csv('webapp/media/{}/{}.csv'.format(word, word), encoding='cp949')

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

        model = RandomForestRegressor()
        model.fit(X_train,y_train)   # 예측할 때는 맨 마지막 종가 예측 결과만 출력해준다

        predicted_stock_price = model.predict(X_train[-7:])  # 과거 7일 어치 예측
        past = back_MinMax(df.iloc[train_size-window:,-2], predicted_stock_price)
        past = np.reshape(past, (-1,))

        currentprice = df["종가"][len(df)-1]
        predictprice = pd.read_csv('webapp/media/predict.csv')
        # print(predictprice)
        predictprice = predictprice[word]
        # print(predictprice)
        predictprice = predictprice.iloc[int(quantity)]
        
        past_real = df["종가"][-7:]
        print(past_real.shape)
        future = pd.read_csv('webapp/media/predict.csv')[word].iloc[:int(quantity)]
        print(future.shape)
        past_tong = pd.concat([past_real, future], axis = 0).reset_index(drop=True)
        print("과거 합쳐", past_tong.shape)
        past_predict = pd.DataFrame(past)
        print("얘가 이상한", past_predict.shape)
        future_tong = pd.concat([past_predict, future], axis = 0).reset_index(drop=True)
        print("미래 합쳐", future_tong.shape)
        df3 = pd.DataFrame(range(0, future_tong.shape[0], 1)).reset_index(drop=True)
        print("날짜", df3.shape)
        df4 = pd.concat([df3, past_tong, future_tong], axis=1)
        print(df4.shape)
        df = df4.values.tolist()

        # print(predictprice)
        surgerate = round(100*((predictprice-currentprice)/currentprice),2)
        if surgerate>0:
            triangle = 1
        else:
            triangle = None
        if surgerate>20:
            opinion= 4
        elif surgerate>10:
            opinion=3
        elif surgerate>-10:
            opinion=2
        elif surgerate>-20:
            opinion=1
        else:
            opinion = 0
        ratio = 0
        
        predictprice = pd.read_csv('webapp/media/predict.csv')[word][:30]
        minimumrange = 0
        for i in range(30):
            for j in range(i, 30):
                if predictprice[j]/predictprice[i]>minimumrange:
                    it = i
                    dt = j
                    minimumrange = predictprice[j]/predictprice[i]
        # print(predictprice)
        # print(it, dt)
        
        it+=1
        dt+=1
        
        showstock = pd.read_csv(os.path.join(os.getcwd(), "webapp", "media", "상장법인목록.csv"), encoding='cp949', index_col=1)
        showstock.index = [format(code, '06') for code in showstock.index]
        showstock = showstock.loc[word]
        return render(request, 'front/stock_detail_dj.html', {'showstock': showstock, 'df':df, 'ratio':ratio, 'it':it, 'dt':dt, 'surgerate':surgerate, 'currentprice':currentprice, 'triangle':triangle ,'opinion':opinion})