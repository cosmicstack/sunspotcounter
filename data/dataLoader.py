import os
import pandas as pd
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class SunSpotDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


if __name__ == "__main__":
    dataset = SunSpotDataset(
        annotations_file="data/img/labels.txt",
        img_dir="data/img/",
        transform=None,
        target_transform=None,
    )

    print(f"Length of dataset: {len(dataset)}")

    train_dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)
    for data in train_dataloader:
        img, label = data
        print(img.shape)
        print(label.shape)

        print(label[0])
        plt.imshow(img[0].permute(1, 2, 0), cmap="gray")
        plt.show()

        break