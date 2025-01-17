from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import resnet18

CURRENT_DIR = Path(__file__).resolve().parent

def get_model():
    model = resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    for item in model.parameters():
        item.requires_grad = False
    
    for item in model.fc.parameters():
        item.requires_grad = True
    
    return model

def load_model(model_name: str = "sunspot_model", with_weights: bool = False, **kwargs):
    model = get_model()
    if with_weights:
        model_name = f"{model_name}.th"
        model_path = CURRENT_DIR / model_name
        model.load_state_dict(torch.load(model_path))
    return model

def save_model(model: torch.nn.Module):
    model_path = CURRENT_DIR / 'sunspot_model.th'
    torch.save(model.state_dict(), model_path)
    return model_path

if __name__ == '__main__':
    model = get_model()
    print(model)