
import torchvision
from torch import nn
from torchvision import transforms

def create_effnetb2_v2_m(num_classes:int=102):
  """
  Creates an EfficientNet V2 M model.
  
  Keyword Arguments:
    :arg num_classes: number of classes
    :type num_classes: int 

  Example Usage:
    model, train_transforms, test_transforms = create_effnetb2_v2_m(102)
  """
  effnetb2_v2_m_weights = torchvision.models.EfficientNet_V2_M_Weights.DEFAULT
  effnetb2_v2_m = torchvision.models.efficientnet_v2_m(weights=effnetb2_v2_m_weights)
  for param in effnetb2_v2_m.parameters():
    param.requires_grad = False

    effnetb2_v2_m.classifier = nn.Sequential(
      nn.Dropout(p=0.3, inplace=True),
      nn.Linear(in_features=1280,
                out_features=num_classes,
                bias=True)
    ) 

  test_transforms = effnetb2_v2_m_weights.transforms()
  train_transforms = torchvision.transforms.Compose([
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    test_transforms
    
  ])

  return effnetb2_v2_m, train_transforms, test_transforms
