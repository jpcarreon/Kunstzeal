import gc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2 as cv
import librosa
from librosa.display import specshow
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from cv2 import imread
from random import randint


class ConvNetD(nn.Module):
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

            return classification[pred]


classification = {
    0: "FLAC", 1: "V0", 2: "320K",
    3: "192K", 4: "128K"
}

# custom colormap to mimic the colors of spek (https://github.com/alexkay/spek)
spek_cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "spek",
    ["black", "indigo", "blue", "cyan", "chartreuse", "yellow", "orange", "red"]
)


def __loadAudioFile(filepath, n_fft, win_size, hop_size, amin, top_db):
    """
        Loads an audio file and returns the data represented in dB and its sampling rate.
        Relies on librosa to perform short-time fourier transform and the conversion of amplitude to dB

        Parameters
        ----
        filepath : str
            path to the audio file
        
        n_fft : int > 0
        
        win_size : int <= n_fft
        
        hop_size : int > 0
        
        amin : float > 0

        top_db : float >= 0

        Returns
        ----
        dB : np.ndarray
            converted amplitude of input audio file

        sr : int
            sampling rate of input audio file
    """

    # specify sr=None to preserve correct sampling rate
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
    outputpath,
    n_fft=2048,
    win_size=2048,
    hop_size=512,
    amin=0.00184,
    top_db=120,
    size_x=2.56,
    size_y=2.56
):
    """
        Takes a single audio file and saves the spectrogram representation to the output path given.
        Relies on librosa to convert the audio file and matplotlib to output the figure.

        Parameters
        ----
        filepath : str
            path to the audio file
        
        outputpath: str
            path to save the generated spectrogram
        
        n_fft : int > 0
        
        win_size : int <= n_fft
        
        hop_size : int > 0
        
        amin : float > 0

        top_db : float >= 0

        size_x : int > 0
            width * 100 of the spectrogram
        
        size_y : int > 0
            height * 100 of the spectrogram

        Returns 
        ----
        None
    """

    dB, sr = __loadAudioFile(filepath, n_fft, win_size, hop_size, amin, top_db)

    fig, ax = plt.subplots(figsize=(size_x, size_y))
    specshow(dB, sr=sr, ax=ax, x_axis="time",
                   y_axis="hz", cmap=spek_cmap)

    plt.axis("off")
    fig.tight_layout(pad=0)
    
    # convert plt figure to numpy 2d array representation of an image
    fig.canvas.draw()
    img_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_data = img_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # convert RGB to BGR before saving
    cv.imwrite(outputpath, cv.cvtColor(img_data, cv.COLOR_RGB2BGR))

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
    """
        Takes a single audio file and displays the spectrogram.
        Relies on librosa to convert the audio file and matplotlib to show the figure.

        Parameters
        ----
        filepath : str
            path to the audio file

        filename: str
            title to display on the figure
            
        n_fft : int > 0
        
        win_size : int <= n_fft
        
        hop_size : int > 0
        
        amin : float > 0

        top_db : float >= 0

        size_x : int > 0
            width * 100 of the spectrogram
        
        size_y : int > 0
            height * 100 of the spectrogram

        Returns 
        ----
        None
    """

    dB, sr = __loadAudioFile(filepath, n_fft, win_size, hop_size, amin, top_db)

    fig, ax = plt.subplots(figsize=(size_x, size_y))
    img = specshow(dB, sr=sr, ax=ax, x_axis="time",
                   y_axis="hz", cmap=spek_cmap)

    fig.colorbar(img, ax=ax, format=f"%0.2f dB")
    fig.tight_layout(pad=2)
    fig.canvas.manager.set_window_title("Spectrogram")
    
    plt.title(filename)
    plt.show(block=True)


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
    """
        Takes a single audio file and converts it to a spectrogram to make a prediction using the given neural network.
        Relies of librosa to convert the audio file and matplotlib to save the spectrogram

        Parameters
        ----
        filepath : str
            path to the audio file

        net : nn.Module()
            neural network architecture to use to predict
        
        n_fft : int > 0
        
        win_size : int <= n_fft
        
        hop_size : int > 0
        
        amin : float > 0

        top_db : float >= 0

        size_x : int > 0
            width * 100 of the spectrogram
        
        size_y : int > 0
            height * 100 of the spectrogram

        Returns 
        ----
        pred : str
            prediction the neural network made from the spectrogram
            possible values: ("FLAC", "V0", "320K", "192K", "128K")
    """
    
    dB, sr = __loadAudioFile(filepath, n_fft, win_size, hop_size, amin, top_db)

    fig, ax = plt.subplots(figsize=(size_x, size_y))
    img = specshow(dB, sr=sr, ax=ax, x_axis="time",
                   y_axis="hz", cmap=spek_cmap)

    plt.axis("off")
    fig.tight_layout(pad=0)

    # convert plt figure to numpy 2d array representation of an image
    fig.canvas.draw()
    img_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_data = img_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # save spectrogram with randomized filename
    outputpath = f"./lib/tmp/{randint(100, 999)}.png"
    
    # convert RGB to BGR before saving
    cv.imwrite(outputpath, cv.cvtColor(img_data, cv.COLOR_RGB2BGR))

    # clear figure and reclaim memory
    plt.clf()
    plt.close('all')
    gc.collect()

    pred = net.predictSingle(outputpath)

    return pred
