import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import resnet34, ResNet34_Weights

# Unzip the dataset
with zipfile.ZipFile("data_sample.zip", "r") as zip_ref:
    total_files = len(zip_ref.infolist())
    for file in tqdm(zip_ref.infolist(), total=total_files):
        zip_ref.extract(file, "data_sample")

# Remove unwanted file
unwanted_file = "data_sample/data_sample/.DS_Store"
if os.path.exists(unwanted_file):
    os.remove(unwanted_file)

# Load the labels
labels_df = pd.read_csv("labels.csv")
print(labels_df.head())

# Visualize sample data
fig = plt.figure(figsize=(25, 8))
path2data = "data_sample/data_sample"
train_imgs = os.listdir(path2data)
for idx, img in enumerate(np.random.choice(train_imgs, 30)):
    ax = fig.add_subplot(3, 30 // 3, idx + 1)
    im = Image.open(os.path.join(path2data, img))
    plt.imshow(im)
    lab = labels_df.loc[labels_df["id"] == img.split('.')[0], 'label'].values[0]
    ax.set_title(f'Label: {lab}')

# Define the custom dataset class
class CancerDataset(Dataset):
    def __init__(self, data_dir, transform, dataset_type=None):
        path2data = os.path.join(data_dir, "data_sample/data_sample")
        filenames = os.listdir(path2data)
        self.full_filenames = [os.path.join(path2data, f) for f in filenames]
        path2labels = os.path.join(data_dir, "labels.csv")
        labels_df = pd.read_csv(path2labels)
        labels_df.set_index("id", inplace=True)

        if dataset_type == "train":
            self.labels = [labels_df.loc[filename[:-4]].values[0] for filename in filenames][:3608]
            self.full_filenames = self.full_filenames[:2608]
        elif dataset_type == "val":
            self.labels = [labels_df.loc[filename[:-4]].values[0] for filename in filenames][3608:3648]
            self.full_filenames = self.full_filenames[3508:3648]
        elif dataset_type == "test":
            self.labels = [labels_df.loc[filename[:-4]].values[0] for filename in filenames][3648:-1]
            self.full_filenames = self.full_filenames[3648:-1]
        else:
            self.labels = [labels_df.loc[filename[:-4]].values[0] for filename in filenames]

        self.transform = transform

    def __len__(self):
        return len(self.full_filenames)

    def __getitem__(self, idx):
        img = Image.open(self.full_filenames[idx])
        img = self.transform(img)
        return img, self.labels[idx]

# Define transformations
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
composed_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=5),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
composed = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Create dataset objects
data_dir = "./"
training_set = CancerDataset(data_dir, transform=composed_train, dataset_type="train")
validation_set = CancerDataset(data_dir, transform=composed, dataset_type="val")
test_set = CancerDataset(data_dir, transform=composed, dataset_type="test")

# Set DataLoader parameters
train_loader = DataLoader(training_set, batch_size=10, shuffle=True, num_workers=0)  # Set num_workers=0 to avoid multiprocessing issues
test_loader = DataLoader(test_set, batch_size=10, shuffle=False, num_workers=0)

# Set hyperparameters
batch_size = 30
lr = 3e-4
n_epochs = 5

# Load the pretrained model and modify it with updated weights parameter
model = resnet34(weights=ResNet34_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training and Testing within if __name__ == "__main__" block to ensure proper multiprocessing on Windows
if __name__ == '__main__':
    # Train the model
    for epoch in range(n_epochs):  # Use more than one epoch if possible
        running_loss = 0.0
        for i, data in enumerate(tqdm(train_loader)):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                tqdm.write(f'Epoch [{epoch + 1}/{n_epochs}], Step [{i + 1}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

    # Test the model
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(test_loader):
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct / total} %')
