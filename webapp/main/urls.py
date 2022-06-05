from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [                 #동일 폴더 내 views에서 html 연결

    path(r'', views.home, name='home'),
    path(r'consult/', views.consult, name='consult'),
    path(r'management/', views.management, name='management'),
    path(r'market/', views.market, name='market'),
    path(r'monitoring/', views.monitoring, name='monitoring'),
    path(r'investment/', views.investment, name='investment'),
]
