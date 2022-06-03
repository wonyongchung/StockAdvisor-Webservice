import pandas as pd
from datetime import datetime, timedelta
from pytz import timezone
from pykrx import stock
import glob, os

time_now = datetime.now(tz=timezone('Asia/Seoul'))
time_now_str = time_now.strftime(format='%Y%m%d')
# ticker_list = stock.get_market_ticker_list(date=time_now, market='KOSPI')
time_start = time_now - timedelta(days=365*5)       #5년전부터 가져오기
time_start_str = time_start.strftime(format='%Y%m%d')

data_dir = os.path.join("..", 'media')

# 전체 주식 시가총액 불러오기
market_cap = stock.get_market_cap(date=time_now_str, prev=True,  market='KOSPI')  #시가총액 순으로 정렬된 dataframe 반환
market_cap['종목이름'] = [stock.get_market_ticker_name(tk) for tk in market_cap.index]
market_cap = market_cap[:50]
# ticker로 5년간 데이터 불러오기

for ticker in market_cap.index:
    
    # ticker_name = stock.get_market_ticker_name(ticker)
    ticker_name = market_cap.loc[ticker, '종목이름']
    ticker_dir = os.path.join(data_dir , f"{ticker_name}")
    try:                                        #회사 폴더 없으면 생성
        os.mkdir(ticker_dir)       
        print(f"Directory Created")
    except:                                     #있으면 말고
        pass

    data = stock.get_market_ohlcv_by_date(fromdate=time_start_str, todate=time_now_str, ticker=ticker)
    data.to_csv(f"{ticker_dir}/{ticker_name}.csv", encoding='cp949')
    
