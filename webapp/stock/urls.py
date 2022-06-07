from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [

    path(r'stock/', views.stock, name='stock'),
    path(r'stock_detail_dj', views.stock_detail_dj, name='stock_detail_dj'),

]
