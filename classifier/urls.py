from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('train/', views.train_model, name='train_model'),
    path('predict/', views.predict, name='predict'),
    #path('get_model_metrics/', views.get_model_metrics, name='get_model_metrics'),
]
