import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNetA(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)

        # Result of 3rd max pooling is 28x28x32 images; thus fc layer has 28*28*32 neurons
        self.fc1 = nn.Linear(30*30*128, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 5)

    def forward(self, x):
        # input data -> 3, 256, 256
        #               channels, width, length
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = torch.flatten(x, 1)               # flatten -> 25088
        x = F.relu(self.fc1(x))               # fc layer 1 -> 128
        x = F.relu(self.fc2(x))               # fc layer 2 -> 128
        x = self.fc3(x)                       # fc layer 3 -> 5
        return x


class ConvNetB(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 64, 3, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1)
        self.conv3 = nn.Conv2d(128, 256, 3, 1)
        self.conv4 = nn.Conv2d(256, 512, 3, 1)

        # Result of 3rd max pooling is 28x28x32 images; thus fc layer has 28*28*32 neurons
        self.fc1 = nn.Linear(14*14*512, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 5)

    def forward(self, x):
        # input data -> 3, 256, 256
        #               channels, width, length
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        x = torch.flatten(x, 1)               # flatten -> 25088
        x = F.relu(self.fc1(x))               # fc layer 1 -> 128
        x = F.relu(self.fc2(x))               # fc layer 2 -> 128
        x = self.fc3(x)                       # fc layer 3 -> 5
        return x


class ConvNetC(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 64, 11, 1)
        self.conv2 = nn.Conv2d(64, 128, 5, 1)
        self.conv3 = nn.Conv2d(128, 256, 3, 1)
        self.conv4 = nn.Conv2d(256, 512, 3, 1)

        # Result of 3rd max pooling is 28x28x32 images; thus fc layer has 28*28*32 neurons
        self.fc1 = nn.Linear(13*13*512, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 5)

    def forward(self, x):
        # input data -> 3, 256, 256
        #               channels, width, length
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        x = torch.flatten(x, 1)               # flatten -> 25088
        x = F.relu(self.fc1(x))               # fc layer 1 -> 128
        x = F.relu(self.fc2(x))               # fc layer 2 -> 128
        x = self.fc3(x)                       # fc layer 3 -> 5
        return x


class ConvNetD(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 64, 5, 1)
        self.conv2 = nn.Conv2d(64, 128, 5, 1)
        self.conv3 = nn.Conv2d(128, 256, 3, 1)
        self.conv4 = nn.Conv2d(256, 512, 3, 1)
        self.conv5 = nn.Conv2d(512, 512, 3, 1)

        # Result of 3rd max pooling is 28x28x32 images; thus fc layer has 28*28*32 neurons
        self.fc1 = nn.Linear(5*5*512, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 5)

    def forward(self, x):
        # input data -> 3, 256, 256
        #               channels, width, length
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))

        x = torch.flatten(x, 1)               # flatten -> 25088
        x = F.relu(self.fc1(x))               # fc layer 1 -> 128
        x = F.relu(self.fc2(x))               # fc layer 2 -> 128
        x = self.fc3(x)                       # fc layer 3 -> 5
        return x


class ConvNetEB(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 64, 3, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1)
        self.conv3 = nn.Conv2d(128, 256, 3, 1)
        self.conv4 = nn.Conv2d(256, 512, 3, 1)
        self.conv5 = nn.Conv2d(512, 1024, 3, 1)

        # Result of 3rd max pooling is 28x28x32 images; thus fc layer has 28*28*32 neurons
        self.fc1 = nn.Linear(6*6*1024, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 5)

    def forward(self, x):
        # input data -> 3, 256, 256
        #               channels, width, length
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))

        x = torch.flatten(x, 1)               # flatten -> 25088
        x = F.relu(self.fc1(x))               # fc layer 1 -> 128
        x = F.relu(self.fc2(x))               # fc layer 2 -> 128
        x = self.fc3(x)                       # fc layer 3 -> 5
        return x


class ConvNetFD(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 64, 5, 1)
        self.conv2 = nn.Conv2d(64, 128, 5, 1)
        self.conv3 = nn.Conv2d(128, 256, 3, 1)
        self.conv4 = nn.Conv2d(256, 512, 3, 1)
        self.conv5 = nn.Conv2d(512, 1024, 3, 1)
        self.conv6 = nn.Conv2d(1024, 1024, 3, 1, padding="same")

        # Result of 3rd max pooling is 28x28x32 images; thus fc layer has 28*28*32 neurons
        self.fc1 = nn.Linear(5*5*1024, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 5)

    def forward(self, x):
        # input data -> 3, 256, 256
        #               channels, width, length
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv6(F.relu(self.conv5(x)))))

        x = torch.flatten(x, 1)               # flatten -> 25088
        x = F.relu(self.fc1(x))               # fc layer 1 -> 128
        x = F.relu(self.fc2(x))               # fc layer 2 -> 128
        x = self.fc3(x)                       # fc layer 3 -> 5
        return x


class ConvNetG(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 64, 3, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1)

        self.conv3 = nn.Conv2d(128, 256, 3, 1)
        self.conv4 = nn.Conv2d(256, 256, 3, 1, padding="same")

        self.conv5 = nn.Conv2d(256, 512, 3, 1)
        self.conv6 = nn.Conv2d(512, 512, 3, 1, padding="same")

        self.conv7 = nn.Conv2d(512, 1024, 3, 1)
        self.conv8 = nn.Conv2d(1024, 1024, 3, 1, padding="same")

        self.fc1 = nn.Linear(6*6*1024, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 5)

    def forward(self, x):
        # input data -> 3, 256, 256
        #               channels, width, length
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = self.pool(F.relu(self.conv6(F.relu(self.conv5(x)))))
        x = self.pool(F.relu(self.conv8(F.relu(self.conv7(x)))))

        x = torch.flatten(x, 1)               # flatten -> 25088
        x = F.relu(self.fc1(x))               # fc layer 1 -> 128
        x = F.relu(self.fc2(x))               # fc layer 2 -> 128
        x = self.fc3(x)                       # fc layer 3 -> 5
        return x


class ConvNetH(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 64, 3, 1, padding="same")
        self.conv2 = nn.Conv2d(64, 64, 3, 1, padding="same")

        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding="same")
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding="same")

        self.conv5 = nn.Conv2d(128, 256, 3, 1, padding="same")
        self.conv6 = nn.Conv2d(256, 256, 3, 1, padding="same")

        self.conv7 = nn.Conv2d(256, 512, 3, 1, padding="same")
        self.conv8 = nn.Conv2d(512, 512, 3, 1, padding="same")

        self.fc1 = nn.Linear(8*8*512, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 5)

    def forward(self, x):
        # input data -> 3, 256, 256
        #               channels, width, length
        x = self.pool(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = self.pool(F.relu(self.conv6(
            F.relu(self.conv6(F.relu(self.conv6(F.relu(self.conv5(x)))))))))
        x = self.pool(F.relu(self.conv8(
            F.relu(self.conv8(F.relu(self.conv8(F.relu(self.conv7(x)))))))))
        x = self.pool(F.relu(self.conv8(
            F.relu(self.conv8(F.relu(self.conv8(F.relu(self.conv8(x)))))))))

        x = torch.flatten(x, 1)               # flatten -> 25088
        x = F.relu(self.fc1(x))               # fc layer 1 -> 128
        x = F.relu(self.fc2(x))               # fc layer 2 -> 128
        x = self.fc3(x)                       # fc layer 3 -> 5
        return x
