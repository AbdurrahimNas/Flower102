from django.urls import path
from Flower102Interface import views


urlpatterns = [
    path("FlowerPrediction", views.ShowPrediction, name="FlowerPrediction")
]
