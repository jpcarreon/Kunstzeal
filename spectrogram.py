import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import librosa
from librosa.display import specshow

class SpectrogramGenerator:
    spek_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "spek",
        ["black", "indigo", "blue", "cyan", "chartreuse", "yellow", "orange", "red"]
    )
    N_FFT = 2048
    WIN_SIZE = 2048
    HOP_SIZE = WIN_SIZE // 4
    AMIN = 0.00184
    TOP_DB = 120

    def __init__(self, filepath):
        self.src = filepath
        self.__generate(filepath)
        

    def __generate(self, filepath):
        amp, sr = librosa.load(filepath, sr = None)
        stft = librosa.stft(
            amp, 
            n_fft = self.N_FFT,
            win_length = self.WIN_SIZE,
            hop_length = self.HOP_SIZE
        )
        dB = librosa.amplitude_to_db(
            np.abs(stft), 
            ref = np.max,
            amin = self.AMIN,
            top_db = self.TOP_DB
        )

        fig, ax = plt.subplots(figsize = (5, 3))
        img = specshow(dB, sr = sr, ax = ax, x_axis = "time", y_axis = "hz", cmap = self.spek_cmap)

        self.fig = fig
        self.img = img
        self.ax = ax
    
    def displaySpectrogram(self):
        plt.axis("on")
        self.fig.colorbar(self.img, ax = self.ax, format = f"%0.2f dB")
        self.fig.tight_layout()
        plt.show()

    def saveSpectrogram(self, filepath):
        plt.axis("off")
        self.fig.tight_layout(pad=0)
        plt.savefig(filepath)
        
