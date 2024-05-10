import torch
import glob as glob
import os
import skimage
import torch.nn as nn
import torchvision
from sklearn.metrics import roc_auc_score
#from model import ResNet18Binary
from torchvision import transforms
from PIL import Image
from io import BytesIO
import base64

IMAGE_SIZE = 224
DEVICE = 'cpu'


def predictm(imgdec):
  
  model = torchvision.models.resnet18(pretrained=True, progress=True)
  model.eval()
  
  model.load_state_dict(torch.load('mhist_pretrained.pth'),strict=False)
  

  transform = transforms.Compose([
  #transforms.ToPILImage(),
  transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
  transforms.ToTensor(),
  transforms.Normalize(
      mean=[0.5, 0.5, 0.5],
      std=[0.5, 0.5, 0.5])
  ])

  
  decoded_img = base64.b64decode(imgdec)
  image = Image.open(BytesIO(decoded_img))
    
  # Preprocess the image. 
  image = transform(image)
  image = torch.unsqueeze(image, 0)
  image = image.to(DEVICE)
    
  # Forward pass throught the image.
  outputs = model(image)
  _, predicted = torch.max(outputs.data, 1)
  

  output_sigmoid = torch.sigmoid(predicted)  

  pred_class_name = 'HP' if predicted > 0.5 else 'SSA'
  resdict = {"predicted":pred_class_name }
  # Annotate the image with ground truth.
  return resdict
