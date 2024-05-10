import torch 
from torch import nn
import torchvision




class ResNet18Binary(nn.Module):
    def __init__(self):
        super(ResNet18Binary, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True, progress=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 2)  # Change the output layer for binary classification
        #self.dropout = nn.Dropout(0.2)
        #resnet.fc



    def forward(self, x):
        #x = self.dropout(x)
        return torch.sigmoid(self.resnet(x))


class DenseNetBinary(nn.Module):
    def __init__(self):
        super(DenseNetBinary,self).__init__()
        self.densenet = torchvision.models.densenet121(pretrained=True, progress=True)
        self.densenet.fc = nn.Linear(self.densenet.classifier.in_features, 2)

    