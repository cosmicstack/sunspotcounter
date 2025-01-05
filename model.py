from pathlib import Path

import torch
import torch.nn as nn

INPUT_MEAN = 0.5
INPUT_STD = 0.5
CURRENT_DIR = Path(__file__).resolve().parent

class SunspotCounter(nn.Module):
    def __init__(self):
        """
        A simple CNN model for counting sunspots in images.
        The model consists of 3 convolutional layers followed by a fully connected layer.

        Args:
            None
        """
        super().__init__()
        self.register_buffer('mean', torch.tensor([INPUT_MEAN]))
        self.register_buffer('std', torch.tensor([INPUT_STD]))

        class ConvBlock(nn.Module):
            def __init__(self, in_channels, out_channels, stride=2, dilation=2):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=dilation, dilation=dilation),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )

            def forward(self, x):
                return self.conv(x)
        
        conv_channels = [16, 32, 64]
        conv_layers = []
        in_channels = 1
        for i in conv_channels:
            conv_layers.append(ConvBlock(in_channels, i))
            in_channels = i
        
        self.conv = nn.Sequential(*conv_layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, height, width)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1)
        """
        x = (x - self.mean) / self.std
        x = self.conv(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x

def save_model(model: torch.nn.Module):
    """
    Save the model to the given path.

    Args:
        model (torch.nn.Module): Model object
        model_path (Path): Path to save the model
    """
    model_path = CURRENT_DIR / 'sunspot_model.th'
    torch.save(model.state_dict(), model_path)
    return model_path

def load_model(model_name="sunspot_model") -> torch.nn.Module:
    """
    Load the model from the given path.

    Args:
        model_path (Path): Path to the model file

    Returns:
        SunspotCounter: Model object
    """
    model = f"{model_name}.th"
    model_path = CURRENT_DIR / model
    model.load_state_dict(torch.load(model_path))
    return model

def debug_model():
    """
    Debug the model by passing a random tensor through it.

    Args:
        None
    """
    model = SunspotCounter()
    x = torch.randn(4, 1, 512, 512) # (b, c, h, w)
    print(f"Input shape: {x.shape}")
    y = model(x)
    print(f"Output shape: {y.shape}")
    print(f"Output Type: {y.dtype}")
    print(f"Output: {y}")
    
if __name__ == "__main__":
    debug_model()