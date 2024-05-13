# MLOPS
MLOPS Projects - This is an comprehensive project.

#  MHIST - Binary Image Classification - Using MLFLOW
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


# Model Training:
  MLFLOW

## Evaluation Metrics


![image](https://github.com/dmahali1983/MLOPS/assets/46201233/e687f878-106b-4b0a-a2ee-d4a26c68f623)

![image](https://github.com/dmahali1983/MLOPS/assets/46201233/f188004a-0db6-47dd-b54c-61572e44d273)
![image](https://github.com/dmahali1983/MLOPS/assets/46201233/ea620ad9-2892-4f80-b079-77a9f2f28708)

![image](https://github.com/dmahali1983/MLOPS/assets/46201233/6bb6362c-0a74-4ff8-8975-4d8d40d2b02a)
![image](https://github.com/dmahali1983/MLOPS/assets/46201233/3b17d248-2669-4b25-8e89-674afdb133f9)
![image](https://github.com/dmahali1983/MLOPS/assets/46201233/5bc5b410-b706-4e48-a15a-526c31c65ae6)
![image](https://github.com/dmahali1983/MLOPS/assets/46201233/a1de31f7-3d67-4167-930d-2a9f7b62039d)

 
-
## Authors

- [@dmahali1983](https://github.com/dmahali1983)
