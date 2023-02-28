import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import librosa
from librosa.display import specshow

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
    size_x=5,
    size_y=3
):
    dB, sr = __loadAudioFile(filepath, n_fft, win_size, hop_size, amin, top_db)

    fig, ax = plt.subplots(figsize=(size_x, size_y))
    img = specshow(dB, sr=sr, ax=ax, x_axis="time", y_axis="hz", cmap=spek_cmap)

    plt.axis("off")
    fig.tight_layout(pad=0)
    plt.savefig(outputPath)
    plt.clf()
    plt.close()


def displaySpectrogram(
    filepath,
    n_fft=2048,
    win_size=2048,
    hop_size=512,
    amin=0.00184,
    top_db=120,
    size_x=5,
    size_y=3
):
    dB, sr = __loadAudioFile(filepath, n_fft, win_size, hop_size, amin, top_db)

    fig, ax = plt.subplots(figsize=(size_x, size_y))
    img = specshow(dB, sr=sr, ax=ax, x_axis="time", y_axis="hz", cmap=spek_cmap)

    fig.colorbar(img, ax=ax, format=f"%0.2f dB")
    fig.tight_layout()
    plt.show()
