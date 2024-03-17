import torch
import matplotlib.pyplot as plt
import torchvision
import model
#from modules import model

def predict(img_path:str,
            model_path:str="./flower102_effnetb2_v2_m.pth"):
  """"
  Predicts on a single image and returns the predicted label.

  Keyword Arguments:
    :arg img_path: Path of the image that would be predicted on.
    :type img_path: str
    :arg model_path: Path of the saved model. Default "./flower102_effnetb2_v2_m.pth"
    :type model_path: str

  Example Usage:
    predicted_label = predict(img_path="./img.jpeg",
                              model_path="./flower102_effnetb2_v2_m.pth")
  """

  device = "cuda" if torch.cuda.is_available() else "cpu"
  effnet_model, _, test_transforms  = model.create_effnetb2_v2_m(102)
  effnet_model.load_state_dict(torch.load(map_location=torch.device(device),f=model_path))
  effnet_model.to(device)

  img = torchvision.io.read_image(img_path)

  img_transformed = test_transforms(img)
 
  with open("./Oxford-102_Flower_dataset_labels.txt", "r") as f:
    class_names= [name for name in f.readlines()]

  effnet_model.eval()
  with torch.inference_mode():
    img_converted = img_transformed.unsqueeze(dim=0)
    img_converted = effnet_model(img_converted.to(device))
    pred_label = torch.argmax(torch.softmax(img_converted, dim=1), dim=1)

  return class_names[pred_label.max()].split("'")[1]
## Push it tomorrow for gibs!!!
