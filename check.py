import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import math
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from datetime import datetime 

# check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Model(nn.Module):
    def _init_(self, n_classes):
        super(Model, self)._init_()
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs
    
# flow()
BATCH_SIZE = 32
N_CLASSES = 10
# trainset, valset, trainloader, valloader = load_data(BATCH_SIZE)
# show_image(trainset, 'MNIST Dataset - preview')
model = Model(N_CLASSES).to(device)
MODEL_NAME = 'model.dth'
model = model.to(device)
model.load_state_dict(torch.load(MODEL_NAME))
# test_model(MODEL_NAME, model, valset, device)