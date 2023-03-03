import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cnn_utils


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 32, 5)        # 1st conv layer: 5x5 kernel with 32 features
        self.conv2 = nn.Conv2d(32, 32, 5)       # 2nd & 3rd conv layers: 5x5 kernel with 32 features
        self.fc1 = nn.Linear(32 * 28 * 28, 128) # Result of 3rd max pooling is 28x28x32 images; thus fc layer has 28*28*32 neurons
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 5)

    def forward(self, x):
        # input data -> 3, 256, 256
        #               channels, width, length
        x = self.pool(F.relu(self.conv1(x)))  # conv layer 1 + mpool -> 32, 126, 126
        x = self.pool(F.relu(self.conv2(x)))  # conv layer 2 + mpool -> 32, 61, 61
        x = self.pool(F.relu(self.conv2(x)))  # conv layer 3 + mpool -> 32, 28, 28
        x = torch.flatten(x, 1)               # flatten -> 25088
        x = F.relu(self.fc1(x))               # fc layer 1 -> 128
        x = F.relu(self.fc2(x))               # fc layer 2 -> 128
        x = self.fc3(x)                       # fc layer 3 -> 5
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 30
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
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    net.parameters(), lr=learning_rate, momentum=momentum)

step_size = ((len(train_set) + 1) // batch_size) // 4

for epoch in range(num_epochs):
    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        inputs = data[0].to(device)
        labels = data[1].to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (i + 1) % step_size == 0:
            print(
                f"[Epoch {epoch + 1}, Step {i + 1:5d}] loss: {running_loss / step_size:.3f}")
            running_loss = 0.0

print("Finished Training")
torch.save(net.state_dict(), "./spectral_net.pt")
