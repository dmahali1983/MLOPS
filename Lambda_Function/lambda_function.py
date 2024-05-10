import json
import urllib.parse
import boto3

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
import pickle
import boto3


IMAGE_SIZE = 224
DEVICE = 'cpu'


#s3 = boto3.client('s3')
#print('Loading function')

def lambda_handler(event, context):
  print("Received event", event)
  AWS_ACCESS_KEY_ID ="AKIAYA6YKLF27N2AWXFN"
  AWS_SECRET_ACCESS_KEY ="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"
  AWS_REGION = "us-east-1"
  s3 = boto3.client('s3', aws_access_key_id= AWS_ACCESS_KEY_ID , aws_secret_access_key = AWS_SECRET_ACCESS_KEY, region_name = AWS_REGION)
  #s3 = boto3.client('s3')

  bucket_name = 'mhistbuck'
  prefix = 'test_data'
  
  response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
  image_objects = [obj for obj in response.get('Contents', []) if obj['Key'].lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
  if not image_objects:
    raise ValueError("No image found in the folder")

        # Find the object with the latest LastModified timestamp
  obj = max(image_objects, key=lambda x: x['LastModified']) 
  
  
  
  image_bytes = s3.get_object(Bucket=bucket_name, Key=obj['Key'])['Body'].read()
 
  image = Image.open(BytesIO(image_bytes))
  
  model = torchvision.models.resnet18(pretrained=False)
  model.eval()

  model.load_state_dict(torch.load('./premodel/mhist_pretrained.pth'),strict=False)
  
  transform = transforms.Compose([
  #transforms.ToPILImage(),
  transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
  transforms.ToTensor(),
  transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5])
  ])
 
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

  print("6666666")
  return {
   'statuscode':200,
   'body':json.dumps(resdict)
  }


# if __name__ == "__main__":
#   event = { "val":"new"}
#   resp = lambda_handler(event, None)
#   print("Response", resp)



