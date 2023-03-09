# Kunstzeal
Kunstzeal is a spectrogram analyzer written in Python. It uses librosa for processing audio and Pyside6 for the GUI. Kunstzeal uses a pre-trained CNN made using PyTorch to classify audio files in one of the following labels:
- FLAC
- V0
- 320K
- 192K
- 128K

The program generates a spectrogram and performs spectrogram analysis my itself. Once finished it will display a judgement on whether the predicted label matches the audio file's metadata. Currently, the model is able to correctly predict 97% of the time.

## Features
- Predicting labels of input audio files
- Displaying Spectrogram of a single audio file

##  License
This software is licensed under the [GNU General Public License v3](LICENSE).