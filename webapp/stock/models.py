from __future__ import unicode_literals
from django.db import models
from django.core.validators import MinLengthValidator
import os
import pandas as pd

class Stock(models.Model):

    # allstocks = pd.read_csv(os.path.join(os.getcwd(), "webapp", "media", "상장법인목록.csv"), encoding='cp949', index_col=0)
    
    # STOCK_CHOICES = [(idx, idx) for idx in allstocks.index]
    
    # name = models.CharField(max_length=30, choices=STOCK_CHOICES)     #드롭다운 메뉴 시도
    name = models.CharField(max_length=30)
    txt = models.TextField(default="-")

    def __str__(self):
        return self.name
class StockDetail(models.Model):

    # allstocks = pd.read_csv(os.path.join(os.getcwd(), "webapp", "media", "상장법인목록.csv"), encoding='cp949', index_col=0)
    
    # STOCK_CHOICES = [(idx, idx) for idx in allstocks.index]
    
    # name = models.ChaerField(max_length=30, choices=STOCK_CHOICES)     #드롭다운 메뉴 시도
    name = models.TextField(max_length=30)
    code = models.TextField(max_length=10, primary_key=True)
    field = models.TextField(max_length=200)
    products = models.TextField(max_length=200)
    list_date = models.TextField(max_length=20)
    closing_month = models.TextField(max_length=20)
    chief = models.TextField(max_length=30)
    pagelink = models.TextField(max_length=200, default='-')
    area = models.TextField(max_length=50)

    def __str__(self):
        return self.name



#상장법인목록 : https://kind.krx.co.kr/corpgeneral/corpList.do?method=loadInitPage
