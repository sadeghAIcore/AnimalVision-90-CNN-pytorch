
import torch
import torchvision
from torch import nn
def create_effnetB2_model(num_of_classes = 90 , 
                          seed = 42):


  # First we download our model EfficientNet_B2_Weights
  weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
  transform = weights.transforms()
  model = torchvision.models.efficientnet_b2(weights=weights)

  # 4. Freeze all layers in base model
  for param in model.parameters():
    param.requires_grad = False

  torch.manual_seed(42)
  torch.cuda.manual_seed(42)
  model.classifier = torch.nn.Sequential(
      nn.Dropout(p=0.3, inplace=True),
      nn.Linear(in_features=1408, out_features=num_of_classes),
  )

  return model , transform
