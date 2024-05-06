# MLOPS
MLOPS Projects


#  MHIST - Binary Image Classification - Using MLFLOW, FLASK API, AWS LAMBDA, SAGE MAKER

This project focuses on analyzing bank marketing data using XGBOOST in AWS SageMaker. The data relates to direct marketing campaigns (phone calls) of a Portuguese banking institution, with the classification goal of predicting whether a client will subscribe to a term deposit.
## Built With

Below listed Tools /Environments used to create, Train, Test and Deploy the model.
 - Python 3.10
 - AWS Boto3
 - AWS Sagemaker Notebook Instance
 - AWS Sagemaker API Endpoint
## Sections

- Data Preparation  [preprocessing.ipynb]
  - Set S3 Bucket and Prefix
  - Get Execution Role
  - Download Data from https://archive.ics.uci.edu/ml/datasets/bank+marketing
  - ETL using Pandas
  - Create the Modeling dataset and upload to S3

- SageMaker Model Training [training.ipynb]
  - Get the SageMaker xgboost Container
  - Set Input Train/Test S3 Path
  - Train the xgboost model using SageMaker Estimator
  
- Model Evaluation and Deployment in SageMaker [Predict.ipynb]
  - Create a prediction API

- Cleaning Up the Environment
## Authors

- [@dmahali1983](https://github.com/dmahali1983)
