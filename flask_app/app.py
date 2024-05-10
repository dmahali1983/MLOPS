import numpy as np
from flask import Flask, request, jsonify, request, flash
from predict import predictm
import os
import sys
from PIL import Image
import io


#def create_app():

app = Flask(__name__)



@app.route('/')
def home():
  return jsonify({"home":'home'})


@app.route('/predict',methods=['POST'])
def predict():    

  if request.method == 'POST':
          
        data = request.json
                  
        encoded_img = data.get('encimg')  
                 
        res = predictm(encoded_img )       
        # delete file after making an inference
        
        #os.remove(saveLocation1)
        # respond with the inference
        rest = {          
        "predclass" : res["predicted"]
        }  
  return jsonify(rest)
  #return app
   
   
   
if __name__ == '__main__':
  app.run( host="0.0.0.0", port=5003)
   

