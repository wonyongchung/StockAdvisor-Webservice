from django.shortcuts import render
from main.models import Main
import numpy as np
import pandas as pd
import os, glob
from pykrx import stock as pystock
from datetime import datetime, timedelta
from pytz import timezone
import random

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

def stock_detail_dj(request):
    if request.method == 'POST':
        word = request.POST.get('stockname').split('-')[-1]

        quantity = request.POST.get('quantity')
        # print(quantity)
        df = pd.read_csv('webapp/media/{}/{}.csv'.format(word, word), encoding='cp949')
        currentprice = df["종가"][len(df)-1]
        predictprice = pd.read_csv('webapp/media/predict.csv')
        # print(predictprice)
        predictprice = predictprice[word]
        # print(predictprice)
        predictprice = predictprice.iloc[int(quantity)]
        
        past_real = df["종가"][-7:]
        #future = pd.read_csv('webapp/media/predict.csv').iloc[:int(quantity)]
        df3 = pd.DataFrame(range(0, 20, 1))
        df4 = pd.concat([df3, past_real], axis=1)
        df = past_real.values.tolist()

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