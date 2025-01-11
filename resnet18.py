from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import resnet18

CURRENT_DIR = Path(__file__).resolve().parent

def get_model():
    pass

def load_model():
    pass

def save_model():
    pass

if __name__ == '__main__':
    model = resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    for item in model.parameters():
        item.requires_grad = False
    
    for item in model.fc.parameters():
        item.requires_grad = True
    
    print(model)