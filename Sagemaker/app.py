import numpy as np
from flask import Flask, request, jsonify, request, flash
from inference import load_model, predict_fun
import sys



#def create_app():

app = Flask(__name__)


@app.route('/')
def home():
  return jsonify({"home":'home'})

@app.route('/ping',methods=['GET'])
def ping():
  """
  Healthcheck function

  """
  return "pong"

@app.route('/invocations',methods=['POST'])
def invocations():  
  print("invoke predict")  

  if request.method == 'POST':
        print("invoke predict Post")  
          
        data = predict_fun(None, "application/json")
        print("data")
                  
        
  return data
  #return app
   
   
   
if __name__ == '__main__':
  app.run( host="0.0.0.0", port=5004)
   

