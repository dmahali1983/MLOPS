# MLOPS
MLOPS Projects - This is an comprehensive project.

#  MHIST - Binary Image Classification - Using MLFLOW, FLASK API, AWS LAMBDA, SAGE MAKER
The objective of the project is to classify the images based on the predominant histological pattern of each image. There are 2 classes to identify the image as benign (HP) and non-benign (SSA).

## Data
Source: https://bmirds.github.io/MHIST/
The dataset includes:
Annotations.csv: It has 3 columns (Image name, Class name, Image Category(train/ test))
Images.zip: It has 3152 images (Including Train and Test Images)

## Algorithm

ResNet18: Residual Network18 is a 72-layer architecture with 18 deep layers.
The primary idea of ResNet is the use of jumping connections that are mostly referred to as shortcut connections or identity connections. These connections primarily function by hopping over one or multiple layers forming shortcuts between these layers. The aim of introducing these shortcut connections was to resolve the predominant issue of vanishing gradient faced by deep networks. These shortcut connections remove the vanishing gradient issue by again using the activations of the previous layer. These identity mappings initially do not do anything much except skip the connections, resulting in the use of previous layer activations. This process of skipping the connection compresses the network; hence, the network learns faster. This compression of the connections is followed by expansion of the layers so that the residual part of the network could also train and explore more feature space.

<img src="images/NLP.png" alt="portfolio img">


## Architecture

<img src="images/NLP.png" alt="portfolio img">


## Below listed Tools /Environments used to create, Train, Test and Deploy the model.

# Model Training:
  MLFLOW

# Model Serve:

 AWS Lambda:
 - Python 3.10
 - AWS Boto3
 - Docker
 - ECR
 - AWS IAM Policy
 - AWS S3
 - AWS Lambda
 - AWS API Gateway Endpoint

  Flask API
  - Python 3.10
  - Flask
  - Docker

  AWS Sagemaker
 - Python 3.10
 - AWS Boto3
 - AWS IAM Policy
 - AWS S3
 - Docker
 - ECR
 - AWS Sagemaker


-
## Authors

- [@dmahali1983](https://github.com/dmahali1983)
