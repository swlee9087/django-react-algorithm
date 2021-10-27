"""admin URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
urlpatterns = [

    path('api/ai_trader/', include('admin.ai_trader.urls')),
    path('api/connect', include('admin.common.urls')),
    path('api/crawling/', include('admin.crawling.urls')),
    path('api/crime/', include('admin.crime.urls')),
    path('api/housing/', include('admin.housing.urls')),
    path('api/iris/', include('admin.iris.urls')),
    path('api/myCNN/', include('admin.myCNN.urls')),
    path('api/myCV2/', include('admin.myCV2.urls')),
    path('api/myGAN/', include('admin.myGAN.urls')),
    path('api/myGRU/', include('admin.myGRU.urls')),
    path('api/myLSTM/', include('admin.myLSTM.urls')),
    path('api/myNLP/', include('admin.myNLP.urls')),
    path('api/myRNN/', include('admin.myRNN.urls')),
    path('api/tensor/', include('admin.tensor.urls')),
    path('api/users/', include('admin.user.urls')),

]
