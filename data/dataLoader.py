import os
import pandas as pd
import torch
from torchvision.transforms import v2
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

class SunSpotDataset(Dataset):
    def __init__(self, annotations_file, img_dir, aug="none"):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.aug = aug
        self.transform = self.get_transform(aug)
    
    def __len__(self):
        return len(self.img_labels)
    
    def get_transform(self, aug):
        if self.aug=="custom":
            return v2.Compose([
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5], std=[0.5])
            ])
        elif self.aug=="pretrained":
            return v2.Compose([
                v2.CenterCrop((480, 480)),
                v2.Grayscale(num_output_channels=3),
                v2.Resize((224, 224)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif self.aug=="none":
            return v2.Compose([
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5], std=[0.5])
            ])

    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = torch.tensor(self.img_labels.iloc[idx, 1], dtype=torch.float32)

        if image.shape[0] not in [1, 3]:
            print(f"Warning: Image {img_path} has {image.shape[0]} channels")
        
        # Convert RGB to grayscale if image has 3 channels
        if image.shape[0] == 3:
            # Use standard RGB to grayscale conversion weights
            # These weights account for human perception of different colors
            weights = torch.tensor([0.2989, 0.5870, 0.1140])
            image = (image.float() * weights.view(-1, 1, 1)).sum(dim=0, keepdim=True)
        elif image.shape[0] != 1:
            raise ValueError(f"Unexpected number of channels: {image.shape[0]} for image {img_path}")

        if self.transform:
            image = self.transform(image)
        
        return image, label


def load_data(
        annotations_file: str,
        img_dir: str,
        num_workers: int,
        batch_size: int,
        aug: str
) -> DataLoader:
    df = pd.read_csv(annotations_file)

    if aug=="custom":
    # Create datasets with different transforms
        train_dataset = SunSpotDataset(annotations_file, img_dir, aug="custom")
        val_dataset = SunSpotDataset(annotations_file, img_dir, aug="none")
        test_dataset = SunSpotDataset(annotations_file, img_dir, aug="none")
    elif aug=="pretrained":
        train_dataset = SunSpotDataset(annotations_file, img_dir, aug="pretrained")
        val_dataset = SunSpotDataset(annotations_file, img_dir, aug="pretrained")
        test_dataset = SunSpotDataset(annotations_file, img_dir, aug="pretrained")
    elif aug=="none":
        train_dataset = SunSpotDataset(annotations_file, img_dir, aug="none")
        val_dataset = SunSpotDataset(annotations_file, img_dir, aug="none")
        test_dataset = SunSpotDataset(annotations_file, img_dir, aug="none")
    
    # Calculate splits
    total_size = len(train_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    
    # Shuffle the DataFrame
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split the DataFrame
    train_df = df_shuffled.iloc[:train_size]
    val_df = df_shuffled.iloc[train_size:train_size+val_size]
    test_df = df_shuffled.iloc[train_size+val_size:]
    
    # Save split DataFrames to temporary files
    train_df.to_csv('data/train_labels.csv', index=False)
    val_df.to_csv('data/val_labels.csv', index=False)
    test_df.to_csv('data/test_labels.csv', index=False)
    
    # Create datasets with different transforms and their respective labels
    if aug=="custom":
        train_dataset = SunSpotDataset('data/train_labels.csv', img_dir, aug="custom")
        val_dataset = SunSpotDataset('data/val_labels.csv', img_dir, aug="none")
        test_dataset = SunSpotDataset('data/test_labels.csv', img_dir, aug="none")
    elif aug=="pretrained":
        train_dataset = SunSpotDataset('data/train_labels.csv', img_dir, aug="pretrained")
        val_dataset = SunSpotDataset('data/val_labels.csv', img_dir, aug="pretrained")
        test_dataset = SunSpotDataset('data/test_labels.csv', img_dir, aug="pretrained")
    elif aug=="none":
        train_dataset = SunSpotDataset('data/train_labels.csv', img_dir, aug="none")
        val_dataset = SunSpotDataset('data/val_labels.csv', img_dir, aug="none")
        test_dataset = SunSpotDataset('data/test_labels.csv', img_dir, aug="none")
    
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    )

if __name__ == "__main__":
    annotations_file = "data/img/labels.txt"
    img_dir = "data/img/"
    num_workers = 2
    batch_size = 32

    train_loader, val_loader, test_loader = load_data(annotations_file, img_dir, num_workers, batch_size, aug="pretrained")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")

    # Display one test image and label
    for images, labels in train_loader:
        print(f"Image batch shape: {images.shape}")
        print(f"Label batch shape: {labels.shape}")
        plt.imshow(images[0].permute(1, 2, 0), cmap='gray')
        plt.title(f"Sunspot count: {labels[0]}")
        plt.show()
        break