import gc
import io
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from librosa.display import specshow
from cv2 import imread


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

    def predictSingle(self, input):
        conv_img = transforms.ToTensor()

        with torch.no_grad():
            img = conv_img(imread(input))
            pred = self(torch.stack([img]))
            pred = torch.max(pred, 1)[1].item()

            return classification[pred]


classification = {
    0: "FLAC", 1: "V0", 2: "320K",
    3: "192K", 4: "128K"
}

spek_cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "spek",
    ["black", "indigo", "blue", "cyan", "chartreuse", "yellow", "orange", "red"]
)


def __loadAudioFile(filepath, n_fft, win_size, hop_size, amin, top_db):
    amp, sr = librosa.load(filepath, sr=None)
    stft = librosa.stft(
        amp,
        n_fft=n_fft,
        win_length=win_size,
        hop_length=hop_size
    )
    dB = librosa.amplitude_to_db(
        np.abs(stft),
        ref=np.max,
        amin=amin,
        top_db=top_db
    )

    return (dB, sr)


def saveSpectrogram(
    filepath,
    outputPath,
    n_fft=2048,
    win_size=2048,
    hop_size=512,
    amin=0.00184,
    top_db=120,
    size_x=2.56,
    size_y=2.56
):
    dB, sr = __loadAudioFile(filepath, n_fft, win_size, hop_size, amin, top_db)

    fig, ax = plt.subplots(figsize=(size_x, size_y))
    img = specshow(dB, sr=sr, ax=ax, x_axis="time",
                   y_axis="hz", cmap=spek_cmap)

    plt.axis("off")
    fig.tight_layout(pad=0)
    plt.savefig(outputPath)

    # clear figure and reclaim memory
    plt.clf()
    plt.close('all')
    gc.collect()


def displaySpectrogram(
    filepath,
    filename,
    n_fft=2048,
    win_size=2048,
    hop_size=512,
    amin=0.00184,
    top_db=120,
    size_x=10,
    size_y=6
):
    dB, sr = __loadAudioFile(filepath, n_fft, win_size, hop_size, amin, top_db)

    fig, ax = plt.subplots(figsize=(size_x, size_y))
    img = specshow(dB, sr=sr, ax=ax, x_axis="time",
                   y_axis="hz", cmap=spek_cmap)

    fig.colorbar(img, ax=ax, format=f"%0.2f dB")
    fig.tight_layout(pad=2)
    fig.canvas.manager.set_window_title("Display Spectrogram")
    plt.title(filename)
    plt.show()


def predictMusic(
    filepath,
    net,
    n_fft=2048,
    win_size=2048,
    hop_size=512,
    amin=0.00184,
    top_db=120,
    size_x=2.56,
    size_y=2.56
):
    dB, sr = __loadAudioFile(filepath, n_fft, win_size, hop_size, amin, top_db)

    fig, ax = plt.subplots(figsize=(size_x, size_y))
    img = specshow(dB, sr=sr, ax=ax, x_axis="time",
                   y_axis="hz", cmap=spek_cmap)

    plt.axis("off")
    fig.tight_layout(pad=0)

    buf = io.BytesIO()
    plt.savefig(buf, format="png")

    # clear figure and reclaim memory
    plt.clf()
    plt.close('all')
    gc.collect()

    with tempfile.NamedTemporaryFile(mode="wb") as fp:
        fp.write(buf.getvalue())

        pred = net.predictSingle(fp.name)
        
    buf.close()

    return pred
