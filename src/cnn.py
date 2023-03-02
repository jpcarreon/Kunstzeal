import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cnn_utils

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        nn.Conv2d(3,6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # -> n, 3, 32, 32
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 14, 14
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 5, 5
        x = torch.flatten(x, 1)               # -> n, 400
        x = F.relu(self.fc1(x))               # -> n, 120
        x = F.relu(self.fc2(x))               # -> n, 84
        x = self.fc3(x)                       # -> n, 10
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 2
batch_size = 4
learning_rate = 0.001
momentum = 0.9

classes = ("FLAC", "V0", "320K", "192K", "128K")

dataset = cnn_utils.SpectrogramDataset(
    "./data/spectral_record.csv", "./data/spectrograms/", transforms.ToTensor())

train_size = int(dataset.__len__() * 0.8)

train_set, test_set = torch.utils.data.random_split(
    dataset, [train_size, dataset.__len__() - train_size])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

# print(cnn_utils.dictCounter([dataset.getItemLabel(i) for i in test_set.indices]))
# cnn_utils.viewRandomSample(train_loader, classes, batch_size)

net = ConvNet().to(device)
