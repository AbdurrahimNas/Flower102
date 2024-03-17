from django.shortcuts import render
from Flower102Interface.predict import predict
from Flower102Interface.forms import FlowerForm
from Flower102Interface.models import Flowers
import os 
# Create your views here.


def ShowPrediction(request):

    with open("Oxford-102_Flower_dataset_labels.txt", "r") as f:
        flower_labels= [name for name in f.readlines()]
        
    nums = range(1, 6)

    if request.method == "POST":
        flowerform = FlowerForm(request.POST, request.FILES)
        if flowerform.is_valid():
            flowerform.save()
            if Flowers.objects.all():
                flower =  Flowers.objects.all()
                flower_data = []
                for obj in flower:
                    flower_data.append({"obj":obj})
            prediction = predict(flower_data[-1]["obj"].flower_image.name, "flower102_effnetb2_v2_m.pth")
            os.remove(flower_data[-1]["obj"].flower_image.name)
              
    else:
        flower_data = None
        flowerform = FlowerForm()
        prediction =  None

    
    return render(request, "Flower102/FlowerPrediction.html", {"flowerform": flowerform,"nums":nums ,"flower_labels":flower_labels , "prediction":prediction})
