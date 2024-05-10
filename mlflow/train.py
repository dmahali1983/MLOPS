import torch
import argparse
import torch.optim as optim
import numpy as np
import sys

import torchvision
from torch import nn
import torch.optim as optim
import mlflow.pytorch
from mlflow import MlflowClient
from urllib.parse import urlparse
from sklearn.metrics import roc_auc_score

from torchvision.transforms import ToTensor

from model import ResNet18Binary
from datasets import (
    get_datasets, get_data_loaders
)
from utils import (
    save_model, save_plots
)

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
# Construct the argument parser.
#parser = argparse.ArgumentParser()
#parser.add_argument(
#    '-e', '--epochs', type=int, default=10,
 #   help='Number of epochs to train our network for'
#)
#parser.add_argument(
 #   '-lr', '--learning-rate', type=float,
  #  dest='learning_rate', default=0.001,
   # help='Learning rate for training the model'
#)
#args = vars(parser.parse_args())

device = ('cuda' if torch.cuda.is_available() else 'cpu')

# Training function.
def train(model, trainloader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    total_loss = 0
    counter = 0
    trainloss = 0.0
    #print("Train ----1")
    for i, batch in enumerate(train_loader,0):
        counter += 1
        inputs = batch['image'].type(torch.FloatTensor).to(device)
        labels = batch['label'].type(torch.LongTensor).to(device)
        #print(inputs)
        #print(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        correct = 0
        total = 0
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        total_loss += loss.item()
        #lr_decay.step()
        running_loss += loss.item()
        if i % 10 == 0 and i != 0:
            print(f'[{epoch}, {i}] loss: {running_loss / 10:.3f}')
            running_loss = 0.0

    
    epoch_loss = total_loss / counter
    #x_traincnt = X_train.shape[0]
    train_acc = 100 * correct // total
    #print("Train ----2")
    return epoch_loss, train_acc, total_loss

def validate(model, val_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    correct_pred = {}
    total_pred = {}
    valloss = []
    total_val_loss = 0.0
    running_loss = 0.0
    counter = 0
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    all_predictions = []
    all_true_labels = []
    with torch.no_grad():
            for i,batch in enumerate(val_loader,0):
                counter += 1
                inputs = batch['image'].type(torch.FloatTensor).to(device)
                labels = batch['label'].type(torch.LongTensor).to(device)
                # calculate outputs by running images through the network
                #start = torch.cuda.Event(enable_timing=True)
                #end = torch.cuda.Event(enable_timing=True)

                #start.record()
                outputs = model(inputs)
                #end.record()
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                running_loss += loss.item()
                #print(f'[{epoch}, {batch}] loss: {running_loss / 10:.3f}')
                if i % 10 == 0 and i != 0:
                  print(f'[{i}, {i}] loss: {running_loss / 10:.3f}')
                  running_loss = 0.0

                # Waits for everything to finish running
                #torch.cuda.synchronize()

                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                true_positives += ((predicted == 1) & (labels == 1)).sum().item()
                true_negatives += ((predicted == 0) & (labels == 0)).sum().item()
                false_positives += ((predicted == 1) & (labels == 0)).sum().item()
                false_negatives += ((predicted == 0) & (labels == 1)).sum().item()

                probabilities = torch.softmax(outputs, dim=1)[:, 1]  # Probability of class 1
                all_predictions.extend(probabilities.cpu().numpy())
                all_true_labels.extend(labels.numpy())

                #predicted = torch.sigmoid(outputs.data)
                #predicted_lb = 1 if predicted > 0.5 else 0
                #print(predicted_lb)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                #print("predicted --> " + clls[predicted_lb] + " Label --> "+ clls[batch['label']] )

            sensitivity = true_positives / (true_positives + false_negatives)
            specificity = true_negatives / (true_negatives + false_positives)
            # Compute ROC-AUC score
            roc_auc = roc_auc_score(all_true_labels, all_predictions)
            print("ROC-AUC Score:", roc_auc)
            #valloss.append(total_val_loss)
            epoch_loss = running_loss / counter
            c_val_acc = 100 * correct // total

            return epoch_loss , c_val_acc, total_val_loss, sensitivity, specificity, roc_auc


if __name__ == '__main__':
    
    # Load the training and validation datasets.
    dataset_train, dataset_valid, dataset_classes = get_datasets()
    
    print(f"[INFO]: Number of training images: {len(dataset_train)}")
    print(f"[INFO]: Number of validation images: {len(dataset_valid)}")
    print(f"[INFO]: Class names: {dataset_classes}\n")
    # Load the training and validation data loaders.
    train_loader, valid_loader = get_data_loaders(dataset_train, dataset_valid)
    
    lr = float(sys.argv[2])
    epochs = int(sys.argv[1])
    model = ResNet18Binary().to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    optimizer = optim.Adam(
        model.parameters(), lr=lr,
        weight_decay=0.005
    )

    criterion = nn.CrossEntropyLoss()

    lr_decay = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.1, last_epoch=-1, verbose=False)


    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    best_val_acc = 0
    loss_plot = []
    val_loss = []
    train_accu = []
    valid_acc = []
    sen = []
    spe = []
    roc = []

with mlflow.start_run() as run:

    # Start the training.
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc, trainloss = train(
            model, train_loader,
            optimizer, criterion, epoch
        )

        
        valid_epoch_loss, valid_epoch_acc, valloss, sensitivity, specificity, roc_auc = validate(model, valid_loader,
                                                    criterion)
        train_loss.append(trainloss)
        valid_loss.append(valloss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        sen.append(sensitivity)
        spe.append(specificity)
        roc.append(roc_auc)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        print(f"LR at end of epoch {epoch+1} {lr_decay.get_last_lr()[0]}")
        print('-'*50)
        mlflow.pytorch.autolog()

        lr_decay.step()

    for i in range(epochs):
        print(str(i))
        print(str(round(train_loss[i])))
        mlflow.log_metric(key ="Training Loss",value=str(train_loss[i]), step = i)
        mlflow.log_metric(key ="Training Accuracy",value=train_acc[i], step = i) 
        mlflow.log_metric(key ="Validation Loss",value=str(valid_loss[i]),step = i)
        mlflow.log_metric(key ="Validation Accuracy",value=valid_acc[i], step = i) 
        mlflow.log_metric(key ="Sensitivity",value=sen[i], step = i) 
        mlflow.log_metric(key ="Specificity",value=spe[i], step = i) 
        mlflow.log_metric(key ="ROC_AUC",value=roc[i], step = i) 


    
    mlflow.pytorch.log_model(model, "model")

    # Save the trained model weights.
    save_model(epochs, model, optimizer, criterion)
    # Save the loss and accuracy plots.
    #save_plots(train_acc, valid_acc, train_loss, valid_loss)
    print('TRAINING COMPLETE')

