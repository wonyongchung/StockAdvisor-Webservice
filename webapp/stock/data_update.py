import pandas as pd
from datetime import datetime, timedelta
from pytz import timezone
from pykrx import stock
import glob, os
from celery import shared_task

@shared_task
def update_data():
    time_now = datetime.now(tz=timezone('Asia/Seoul'))
    time_now_str = time_now.strftime(format='%Y%m%d')
    # ticker_list = stock.get_market_ticker_list(date=time_now, market='KOSPI')
    time_start = time_now - timedelta(days=365*5)       #5년전부터 가져오기
    time_start_str = time_start.strftime(format='%Y%m%d')

    data_dir = os.path.join("..", 'media')

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
        
        # ticker_name = stock.get_market_ticker_name(ticker)
        # ticker_name = market_cap.loc[ticker, '종목이름']
        ticker_dir = os.path.join(data_dir , f"{ticker}")
        # print(ticker)
        
        try:                                        #회사 폴더 없으면 생성
            os.mkdir(ticker_dir)       
        except:                                     #있으면 말고
            pass

        data = stock.get_market_ohlcv_by_date(fromdate=time_start_str, todate=time_now_str, ticker=ticker)
        data.to_csv(f"{ticker_dir}/{ticker}.csv", encoding='cp949')
