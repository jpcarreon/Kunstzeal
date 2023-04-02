import gc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2 as cv
import librosa
from librosa.display import specshow

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
    specshow(dB, sr=sr, ax=ax, x_axis="time", y_axis="hz", cmap=spek_cmap)

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
    img = specshow(dB, sr=sr, ax=ax, x_axis="time", y_axis="hz", cmap=spek_cmap)

    fig.colorbar(img, ax=ax, format=f"%0.2f dB")
    fig.tight_layout()
    plt.show()
