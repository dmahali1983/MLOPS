
from torchvision import datasets, transforms
import skimage
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import sys
from imblearn.over_sampling import SMOTE
import numpy as np
import random
import torch

# Required constants.
ROOT_DIR = 'content'
VALID_SPLIT = 0.20
IMAGE_SIZE = 224 # Image size of resize when applying transforms.
BATCH_SIZE = 32
NUM_WORKERS = 4 # Number of parallel processes for data preparation.
LR = 0.01
EPOCHS = 15
seed = 234
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)





# Training transforms
def get_train_transform():
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),        
        transforms.RandomRotation(25),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
            )
    ])
    return train_transform

# Validation transforms
def get_valid_transform():
    valid_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
            )
    ])
    return valid_transform
	
def undersampling(datasetus, n_classes, n_per_class):
  under_samped_df = pd.DataFrame()
  for i in datasetus['mvl'].unique():
    under_samped_df = pd.concat([under_samped_df, datasetus[datasetus.mvl == i].iloc[:n_per_class,:]])
  return under_samped_df

def smote_sample(datasetus):
    smt = SMOTE(sampling_strategy = 'auto',random_state=27, k_neighbors = 5)

    under_sampled_df = pd.DataFrame()
    X = datasetus['filepath']
    y = datasetus['mvl']
    print(X)
    print(y.value_counts())
    smt_x, smt_y = SMOTE.fit_resample(X,y )
    print(smt_x)
    under_samped_df = pd.concat([smt_x, smt_y], axis =1)
    return under_sampled_df

  
def get_datasets():
    """
    Function to prepare the Datasets.
    :param pretrained: Boolean, True or False.
    Returns the training and validation datasets along
    with the class names.
    """

    df = pd.read_csv("Training_Dataset/annotations_train.csv")
    filenames = dict([(imn,[cat,mvl]) for imn,cat,mvl in zip(df["Image Name"],df.Partition,df["Majority Vote Label"])])
    files = []
    categories = []
    mvl = []

    for keys, val in filenames.items():
      files.append("Training_Dataset/images/"+keys)
      categories.append(val[0])
      mvl.append(val[1])

    df_cmb = pd.DataFrame({'filepath':files,
                  'category': categories,
                  'mvl':mvl
                  })
    #df_train = df_cmb[df_cmb['category']=='train']
    df_train = df_cmb
    #df_test = df_cmb[df_cmb['category']=='test']
    df_train.drop(['category'],axis=1 , inplace= True)
    #df_test.drop(['category'],axis=1 , inplace= True)
    dataset_train = undersampling(df_train, 2, 600)
    
    label_encoder = LabelEncoder()
    dataset_train['mvl'] = label_encoder.fit_transform(dataset_train['mvl'])
    
    X = dataset_train    
    y = dataset_train['mvl']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VALID_SPLIT, stratify=y)
    dataset_train = X_train
    dataset_valid = X_val
    return dataset_train, dataset_valid, y_train
	
class pytorch_data(Dataset):  # // https://www.kaggle.com/code/shtrausslearning/pytorch-cnn-binary-image-classification

  def __init__(self, Dataset, sc="Train"):

    self.labels = Dataset["mvl"]
    self.filepaths = Dataset["filepath"]
    if sc=="Train":
      self.transform = get_train_transform()
    else:
      self.transform = get_valid_transform()

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):

     image_path = self.filepaths.iloc[idx]
     #print(image_path)
     image = skimage.io.imread(image_path) #Image.open(self.filepaths[idx])  # Open Image with PIL
     label = self.labels.iloc[idx]
     #print("label ===>" + str(label))
     image = self.transform(image) # Apply Specific Transformation to Image
     out = {"image":image, "label": label, "img_path":image_path}


     return out
     
     
def get_data_loaders(dataset_train, dataset_valid):
    """
    Prepares the training and validation data loaders.

    :param dataset_train: The training dataset.
    :param dataset_valid: The validation dataset.

    Returns the training and validation data loaders.
    """
    train_loader = DataLoader(
        pytorch_data(dataset_train,"Train" ), batch_size=64

    )
    valid_loader = DataLoader(
        pytorch_data(dataset_valid,"Valid"), batch_size=15
    )
    return train_loader, valid_loader