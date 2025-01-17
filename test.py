import numpy as np
import torch

from resnet18 import load_model
from data.dataLoader import load_data

def test(seed: int = 2025):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps") # for Arm Macs
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    model = load_model(with_weights=True)
    model = model.to(device)

    torch.manual_seed(seed)
    np.random.seed(seed)

    test_data = load_data("data/img/labels.txt", "data/img/", num_workers=2, batch_size=32, aug="pretrained")[2]

    loss_func = torch.nn.MSELoss()

    test_batch_loss = []

    model.eval()
    for images, labels in test_data:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss = loss_func(output.squeeze(1), labels)
        test_batch_loss.append(loss.item())
    
    test_loss = np.mean(test_batch_loss)
    print(f"Test Loss: {test_loss}")

if __name__ == "__main__":
    test()