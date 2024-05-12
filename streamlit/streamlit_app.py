import streamlit as st
from PIL import Image
import requests
import tempfile
import base64
import io
# add title to your app
import os
import boto3

#FLASK_API_URL = "http://127.0.0.1:5000/predict"

S3_BUCKET = "mhistbuck"

def upload_to_s3(imagedt, bucketname, imagenm):
  AWS_ACCESS_KEY_ID =******************
  AWS_SECRET_ACCESS_KEY =******************
  AWS_REGION = "us-east-1"
  s3 = boto3.client('s3', aws_access_key_id= AWS_ACCESS_KEY_ID , aws_secret_access_key = AWS_SECRET_ACCESS_KEY, region_name = AWS_REGION)
  #s3.put_object(Body=imagedt, Bucket = bucketname, Key = imagenm, ContentType = "image/png")
  s3.upload_fileobj(imagedt,bucketname,imagenm, ExtraArgs={"ContentType" : "image/png"})
  st.write("Uploaded image to s3 bucket")
  return "success"


def main():

  FLASK_API_URL = "http://flask_app:5003/predict"

  st.title("MHIST Image Classification")
  l_img = st.file_uploader(label="Upload image for prediction", type=["png","jpg"])

  if l_img is not None:
    uploaded_image = Image.open(l_img)
    st.image(uploaded_image)  
    buffered = io.BytesIO()
    uploaded_image.save(buffered, format="PNG") 
    #encoded_image = base64.b64encode(l_img.read()).decode('utf-8')
    encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')   
    gt_class_name = l_img.name.split('.')[0].split('_')[2]
    
  sl_op = st.selectbox("Prediction Flavor", ("SAGE MAKER","FLASK API","API Gateway", "Grafana" ))

  if sl_op == "API Gateway":
    if st.button("Upload to S3 Bucket"):
        #imgdata = l_img.read()
        st.write("Path  "+l_img.name)
        l_img.seek(0)
        upload_to_s3(l_img,S3_BUCKET, "test_data/test.png")


  if st.button('Predict'):    
    if sl_op == "SAGE Maker":
      st.write(sl_op , "SM")
    elif sl_op == "FLASK API":
      print(FLASK_API_URL)
      response = requests.post(FLASK_API_URL,json={"encimg":encoded_image})
      if response.status_code == 200:
         ress = response.json()
         st.write("Original Class: " + gt_class_name)
         st.write("Predicted Class: " + ress["predclass"])
         
      else:
         st.error(f"Failed to get prediction using flask api.")
      
    elif sl_op == "API Gateway": 
      
      AWS_API_URL = "https://ern8qmvqk9.execute-api.us-east-1.amazonaws.com/prod"
      response = requests.post(AWS_API_URL)
      #print("Response Code --->"+response.status_code )
      if response.status_code == 200:
         ress = response.json()         
         st.write("Original Class: " + gt_class_name)
         st.write("Predicted Class: " + ress["body"])
    else:
     st.write(sl_op , "GF")  


if __name__ == "__main__":
  main()






