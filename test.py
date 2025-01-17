import numpy as np
import matplotlib.pyplot as plt
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
    data_store_op = []
    data_store_label = []
    for images, labels in test_data:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss = loss_func(output.squeeze(1), labels)
        test_batch_loss.append(loss.item())
        data_store_op.append(output)
        data_store_label.append(labels)
    
    test_loss = np.mean(test_batch_loss)
    print(f"Test Loss: {test_loss}")
    
    data_store_op = [x.detach().cpu().numpy() for y in data_store_op for x in y]
    data_store_label = [x.detach().cpu().numpy() for y in data_store_label for x in y]

    min_val = min(data_store_label)
    max_val = max(data_store_label)
    
    plt.figure(figsize=(16, 8))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linewidth=2, linestyle='--', alpha=0.5)
    plt.scatter(data_store_label, data_store_op, color='k', facecolor='white', alpha=0.5)
    plt.xlabel("True Value")
    plt.ylabel("Predicted Value")
    plt.title("True vs Predicted Sunspot Counts")
    plt.savefig("true_vs_pred.png")

if __name__ == "__main__":
    test()