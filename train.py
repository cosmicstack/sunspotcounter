import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from model import load_model, save_model
from data.dataLoader import load_data

def train(
        exp_dir: str = "logs",
        model_name: str = "sunspot_model",
        num_epoch: int = 50,
        lr: float = 1e-3,
        batch_size: int = 32,
        seed: int = 2025,
        **kwargs,
):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")
    
    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    model = load_model()
    logger.add_graph(model, torch.zeros(1, 1, 512, 512))
    model = model.to(device)

    train_data, val_data, test_data = load_data("data/img/labels.txt", "data/img/", num_workers=2, batch_size=batch_size)

    # log first 32 images to tensorboard
    for img, label in train_data:
        logger.add_images("train/images", img[:32])
    
    logger.flush()

    # Create loss function and optimizer
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0

    # Training loop
    for epoch in range(num_epoch):
        metrics = {"train_loss": [], "val_loss": []}
        model.train()
        for images, labels in train_data:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss = loss_func(outputs.squeeze(1), labels)
            logger.add_scalar("train/batch loss", loss.item(), epoch * len(train_data) + global_step)
            metrics["train_loss"].append(loss.item())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

        # Validation
        model.eval()
        for images, labels in val_data:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            loss = loss_func(outputs.squeeze(1), labels)

            metrics["val_loss"].append(loss.item())
        
        # Log metrics
        epoch_train_acc = np.mean(metrics["train_loss"])
        epoch_val_acc = np.mean(metrics["val_loss"])

        logger.add_scalar("train/loss", epoch_train_acc, epoch)
        logger.add_scalar("val/loss", epoch_val_acc, epoch)
        logger.flush()

        # Print first, last and every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train loss={epoch_train_acc:.4f}; "
                f"val loss={epoch_val_acc:.4f}"
            )
        
    # Save model
    save_model(model)

    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the sunspot model")
    parser.add_argument("--num_epoch", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=7000, help="Batch size")
    args = parser.parse_args()

    train(**vars(args))