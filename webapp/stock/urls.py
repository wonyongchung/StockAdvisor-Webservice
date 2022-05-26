from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [

    path(r'stock/', views.stock, name='stock'),
    path(r'stock/(?P<word>)/', views.stock_detail, name='stock_detail'),

]
