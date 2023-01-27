import torch
from PIL import Image
from torchvision import transforms
import requests
from typing import Tuple, Union, Any
import torch.nn.functional as F

#TODO change type Any to proper type

def class_img(model: Any, image_path: str) -> Tuple[str, Union[int, float]]:
  
  transform = transforms.Compose([            
 
    transforms.Resize(256),                    
    transforms.CenterCrop(224),                
    transforms.ToTensor(),                     
    transforms.Normalize(                      
    mean=[0.485, 0.456, 0.406],                
    std=[0.229, 0.224, 0.225]                  
  
  )])

  img = Image.open(image_path)
  batch_t = torch.unsqueeze(transform(img), 0)
  out = model(batch_t)
  
  _, index = torch.max(out, 1)
  
  labels = requests.get('https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt').text.split('\n')
  percentage = F.softmax(out, dim=1)[0] * 100
  
  return (labels[index[0]], percentage[index[0]].item())
