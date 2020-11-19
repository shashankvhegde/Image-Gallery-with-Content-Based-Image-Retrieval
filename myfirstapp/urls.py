from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.image_list, name = 'image_list'),
    path('insert_picture/', views.insert_picture, name = 'insert_picture'),
]