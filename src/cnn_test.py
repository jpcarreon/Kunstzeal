import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
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

def get_all_preds(net, loader):
    all_preds = torch.tensor([])

    for batch in loader:
        images, labels = batch

        preds = net(images)
        all_preds = torch.cat(
            (all_preds, preds),
            dim=0
        )
    
    return all_preds

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4,
                                         shuffle=False, num_workers=2)

net = ConvNet()
net.load_state_dict(torch.load("./cifar_net.pt"))

with torch.no_grad():
    train_preds = get_all_preds(net, test_loader)

    stacked = torch.stack(
        (
            torch.tensor(test_dataset.targets),
            torch.max(train_preds, 1)[1]
        ),
        dim=1
    )

    cm = torch.zeros(10, 10, dtype=torch.int32)

    for p in stacked:
        j, k = p.tolist()
        cm[j, k] += 1
    
    df_cm = pd.DataFrame(cm.tolist(), index=[i for i in classes], columns=[i for i in classes])
    
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, fmt="3d", cmap="rocket_r")

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    plt.show()
    
