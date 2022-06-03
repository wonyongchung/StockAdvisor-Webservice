from __future__ import unicode_literals
from django.db import models
import os
import pandas as pd

class Stock(models.Model):

    allstocks = pd.read_csv(os.path.join(os.getcwd(), "webapp", "media", "상장법인목록.csv"), encoding='cp949', index_col=0)
    
    STOCK_CHOICES = [(idx, idx) for idx in allstocks.index]
    
    # name = models.CharField(max_length=30, choices=STOCK_CHOICES)     #드롭다운 메뉴 시도
    name = models.CharField(max_length=30)
    txt = models.TextField(default="-")

    def __str__(self):
        return self.name



#상장법인목록 : https://kind.krx.co.kr/corpgeneral/corpList.do?method=loadInitPage
