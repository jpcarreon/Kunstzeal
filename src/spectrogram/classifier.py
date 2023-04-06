"""
    Contains all CNN architectures created and tested.

    Sorted according to accuracy:
    1. D
    2. B
    3. FD
    4. EB
    5. C
    6. A
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from cv2 import imread


class nnInference:
    # classification classes for the CNN
    classification = {
        0: "FLAC", 1: "V0", 2: "320K",
        3: "192K", 4: "128K"
    }

    def predictSingle(self, input):
        """
            Perform a single prediction from a given image input.

            Parameters
            ----
            input : str
                path to a spectrogram image
            
            Returns
            ----
            pred : str

        """

        conv_img = transforms.ToTensor()

        with torch.no_grad():
            img = conv_img(imread(input))
            
            # put image in a single batch to predict
            pred = self(torch.stack([img]))
            pred = torch.max(pred, 1)[1].item()

            return self.classification[pred]
        

class ConvNetA(nn.Module, nnInference):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 32, 3, 1)         # 3x3 kernel with stride of 1; Outputs 32 features
        self.conv2 = nn.Conv2d(32, 64, 3, 1)        # 3x3 kernel with stride of 1; Outputs 64 features
        self.conv3 = nn.Conv2d(64, 128, 3, 1)       # 3x3 kernel with stride of 1; Outputs 128 features

        # Result of last max pooling is 30*30*128 images; thus fc1 layer accepts 30*30*128 neurons
        self.fc1 = nn.Linear(30*30*128, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 5)

    def forward(self, x):
        # input data -> 3, 256, 256
        #               channels, width, length
        x = self.pool(F.relu(self.conv1(x)))    # input -> maxpool(conv1(x)) : 127x127x32
        x = self.pool(F.relu(self.conv2(x)))    # input -> maxpool(conv2(x)) : 62x62x64
        x = self.pool(F.relu(self.conv3(x)))    # input -> maxpool(conv3(x)) : 30x30x128

        x = torch.flatten(x, 1)               # flatten -> 115200
        x = F.relu(self.fc1(x))               # fc layer 1 -> 256
        x = F.relu(self.fc2(x))               # fc layer 2 -> 256
        x = self.fc3(x)                       # fc layer 3 -> 5
        return x


class ConvNetB(nn.Module, nnInference):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 64, 3, 1)         # 3x3 kernel with stride of 1; Outputs 64 features
        self.conv2 = nn.Conv2d(64, 128, 3, 1)       # 3x3 kernel with stride of 1; Outputs 128 features
        self.conv3 = nn.Conv2d(128, 256, 3, 1)      # 3x3 kernel with stride of 1; Outputs 256 features
        self.conv4 = nn.Conv2d(256, 512, 3, 1)      # 3x3 kernel with stride of 1; Outputs 512 features

        # Result of last max pooling is 14*14*512 images; thus fc1 layer accepts 14*14*512 neurons
        self.fc1 = nn.Linear(14*14*512, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 5)

    def forward(self, x):
        # input data -> 3, 256, 256
        #               channels, width, length
        x = self.pool(F.relu(self.conv1(x)))        # input -> maxpool(conv1(x)) : 127x127x64
        x = self.pool(F.relu(self.conv2(x)))        # input -> maxpool(conv2(x)) : 62x62x128
        x = self.pool(F.relu(self.conv3(x)))        # input -> maxpool(conv3(x)) : 30x30x256
        x = self.pool(F.relu(self.conv4(x)))        # input -> maxpool(conv4(x)) : 14x14x512

        x = torch.flatten(x, 1)               # flatten -> 100352
        x = F.relu(self.fc1(x))               # fc layer 1 -> 256
        x = F.relu(self.fc2(x))               # fc layer 2 -> 256
        x = self.fc3(x)                       # fc layer 3 -> 5
        return x


class ConvNetC(nn.Module, nnInference):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 64, 11, 1)        # 11x11 kernel with stride of 1; Outputs 64 features
        self.conv2 = nn.Conv2d(64, 128, 5, 1)       # 5x5 kernel with stride of 1; Outputs 128 features
        self.conv3 = nn.Conv2d(128, 256, 3, 1)      # 3x3 kernel with stride of 1; Outputs 256 features
        self.conv4 = nn.Conv2d(256, 512, 3, 1)      # 3x3 kernel with stride of 1; Outputs 512 features

        # Result of last max pooling is 13*13*512 images; thus fc layer accepts 13*13*512 neurons
        self.fc1 = nn.Linear(13*13*512, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 5)

    def forward(self, x):
        # input data -> 3, 256, 256
        #               channels, width, length
        x = self.pool(F.relu(self.conv1(x)))        # input -> maxpool(conv1(x)) : 123x123x64
        x = self.pool(F.relu(self.conv2(x)))        # input -> maxpool(conv2(x)) : 59x59x128
        x = self.pool(F.relu(self.conv3(x)))        # input -> maxpool(conv3(x)) : 28x28x256
        x = self.pool(F.relu(self.conv4(x)))        # input -> maxpool(conv4(x)) : 13x13x512

        x = torch.flatten(x, 1)               # flatten -> 86528
        x = F.relu(self.fc1(x))               # fc layer 1 -> 256
        x = F.relu(self.fc2(x))               # fc layer 2 -> 256
        x = self.fc3(x)                       # fc layer 3 -> 5
        return x


class ConvNetD(nn.Module, nnInference):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 64, 5, 1)         # 5x5 kernel with stride of 1; Outputs 64 features
        self.conv2 = nn.Conv2d(64, 128, 5, 1)       # 5x5 kernel with stride of 1; Outputs 128 features
        self.conv3 = nn.Conv2d(128, 256, 3, 1)      # 3x3 kernel with stride of 1; Outputs 256 features
        self.conv4 = nn.Conv2d(256, 512, 3, 1)      # 3x3 kernel with stride of 1; Outputs 512 features
        self.conv5 = nn.Conv2d(512, 512, 3, 1)      # 3x3 kernel with stride of 1; Outputs 512 features

        # Result of last max pooling is 5*5*512 images; thus fc layer accepts 5*5*512 neurons
        self.fc1 = nn.Linear(5*5*512, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 5)

    def forward(self, x):
        # input data -> 3, 256, 256
        #               channels, width, length
        x = self.pool(F.relu(self.conv1(x)))        # input -> maxpool(conv1(x)) : 126x126x64
        x = self.pool(F.relu(self.conv2(x)))        # input -> maxpool(conv2(x)) : 61x61x128
        x = self.pool(F.relu(self.conv3(x)))        # input -> maxpool(conv3(x)) : 29x29x256
        x = self.pool(F.relu(self.conv4(x)))        # input -> maxpool(conv4(x)) : 13x13x512
        x = self.pool(F.relu(self.conv5(x)))        # input -> maxpool(conv5(x)) : 5x5x512

        x = torch.flatten(x, 1)               # flatten -> 12800
        x = F.relu(self.fc1(x))               # fc layer 1 -> 256
        x = F.relu(self.fc2(x))               # fc layer 2 -> 256
        x = self.fc3(x)                       # fc layer 3 -> 5
        return x


class ConvNetEB(nn.Module, nnInference):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 64, 3, 1)         # 3x3 kernel with stride of 1; Outputs 64 features
        self.conv2 = nn.Conv2d(64, 128, 3, 1)       # 3x3 kernel with stride of 1; Outputs 128 features
        self.conv3 = nn.Conv2d(128, 256, 3, 1)      # 3x3 kernel with stride of 1; Outputs 256 features
        self.conv4 = nn.Conv2d(256, 512, 3, 1)      # 3x3 kernel with stride of 1; Outputs 512 features
        self.conv5 = nn.Conv2d(512, 1024, 3, 1)     # 3x3 kernel with stride of 1; Outputs 1024 features

        # Result of last max pooling is 6*6*1024 images; thus fc layer accepts 6*6*1024 neurons
        self.fc1 = nn.Linear(6*6*1024, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 5)

    def forward(self, x):
        # input data -> 3, 256, 256
        #               channels, width, length
        x = self.pool(F.relu(self.conv1(x)))        # input -> maxpool(conv1(x)) : 127x127x64
        x = self.pool(F.relu(self.conv2(x)))        # input -> maxpool(conv2(x)) : 62x62x128
        x = self.pool(F.relu(self.conv3(x)))        # input -> maxpool(conv3(x)) : 30x30x256
        x = self.pool(F.relu(self.conv4(x)))        # input -> maxpool(conv4(x)) : 14x14x512
        x = self.pool(F.relu(self.conv5(x)))        # input -> maxpool(conv5(x)) : 6x6x1024

        x = torch.flatten(x, 1)               # flatten -> 36864
        x = F.relu(self.fc1(x))               # fc layer 1 -> 256
        x = F.relu(self.fc2(x))               # fc layer 2 -> 256
        x = self.fc3(x)                       # fc layer 3 -> 5
        return x


class ConvNetFD(nn.Module, nnInference):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 64, 5, 1)         # 5x5 kernel with stride of 1; Outputs 64 features
        self.conv2 = nn.Conv2d(64, 128, 5, 1)       # 5x5 kernel with stride of 1; Outputs 128 features
        self.conv3 = nn.Conv2d(128, 256, 3, 1)      # 3x3 kernel with stride of 1; Outputs 256 features
        self.conv4 = nn.Conv2d(256, 512, 3, 1)      # 3x3 kernel with stride of 1; Outputs 512 features
        self.conv5 = nn.Conv2d(512, 1024, 3, 1)     # 3x3 kernel with stride of 1; Outputs 1024 features
        self.conv6 = nn.Conv2d(1024, 1024, 3, 1, padding="same")    # 3x3 kernel with stride of 1 and padding of 1; Outputs 64 features

        # Result of last max pooling is 5*5*1024 images; thus fc layer accepts 5*5*1024 neurons
        self.fc1 = nn.Linear(5*5*1024, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 5)

    def forward(self, x):
        # input data -> 3, 256, 256
        #               channels, width, length
        x = self.pool(F.relu(self.conv1(x)))        # input -> maxpool(conv1(x)) : 126x126x64
        x = self.pool(F.relu(self.conv2(x)))        # input -> maxpool(conv2(x)) : 61x61x128
        x = self.pool(F.relu(self.conv3(x)))        # input -> maxpool(conv3(x)) : 29x29x256
        x = self.pool(F.relu(self.conv4(x)))        # input -> maxpool(conv4(x)) : 13x13x512
        x = self.pool(F.relu(self.conv6(F.relu(self.conv5(x)))))    # input -> maxpool(conv6(conv5(x))) : 5x5x1024

        x = torch.flatten(x, 1)               # flatten -> 25600
        x = F.relu(self.fc1(x))               # fc layer 1 -> 256
        x = F.relu(self.fc2(x))               # fc layer 2 -> 256
        x = self.fc3(x)                       # fc layer 3 -> 5
        return x


class ConvNetG(nn.Module, nnInference):
    # Attempt to use padding="same" for multiple layers; failed to learn
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

        x = torch.flatten(x, 1)               # flatten -> 36864
        x = F.relu(self.fc1(x))               # fc layer 1 -> 256
        x = F.relu(self.fc2(x))               # fc layer 2 -> 256
        x = self.fc3(x)                       # fc layer 3 -> 5
        return x


class ConvNetH(nn.Module, nnInference):
    # based on VGG19 architecture; failed to learn
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

        x = torch.flatten(x, 1)               # flatten -> 32768
        x = F.relu(self.fc1(x))               # fc layer 1 -> 256
        x = F.relu(self.fc2(x))               # fc layer 2 -> 256
        x = self.fc3(x)                       # fc layer 3 -> 5
        return x
