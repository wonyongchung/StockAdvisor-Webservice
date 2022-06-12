from django.shortcuts import render
from main.models import Main
import random

def home(request):

    rand1 = random.randint(0,1)
    print(rand1)
    return render(request, 'front/home.html', {'rand1': rand1})


def consult(request):

    return render(request, 'front/consult.html')


def market(request):

    return render(request, 'front/market.html')


def monitoring(request):

    return render(request, 'front/monitoring.html')


def investment(request):

    return render(request, 'front/investment.html')


def management(request):

    return render(request, 'front/management.html')