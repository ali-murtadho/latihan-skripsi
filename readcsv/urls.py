"""modelinput URL Configuration

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
from django.urls import path
from . import views
urlpatterns = [
    path('data', views.show_data, name='show_data'),
    path('train', views.training, name='train'),
    path('eval', views.evaluation, name='eval'),
    path('testing', views.testing, name='testing'),
    path('classification', views.classification, name='classification'),
    path('prediction', views.prediction, name='prediction'),
    path('importcsv', views.importCsv, name='import_csv'),
    path('importExcel', views.importExcel, name='import_excel'),
    path('register', views.register, name='register'),
    path('login', views.login_view, name='login'),
    path('logout', views.logout_view, name='logout'),
]
