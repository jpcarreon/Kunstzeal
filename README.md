# Kunstzeal
Kunstzeal is a spectrogram analyzer written in Python. It uses librosa for processing audio and Pyside6 for the GUI. Kunstzeal uses a pre-trained CNN made using PyTorch to classify audio files in one of the following labels:
- FLAC
- V0
- 320K
- 192K
- 128K

The program generates a spectrogram and performs spectrogram analysis by itself. Once finished, it will display a judgement on whether the predicted label matches the audio file's metadata. Currently, the model is able to correctly predict 97% of the time.

## Usage
- Add audio files by dragging and dropping `.mp3` or `.flac` files into the program or using the file picker.
- Press the _predict_ button to initiate automated labeling.
- Right-click an added audio file to view the context menu.

##  License
This software is licensed under the [GNU General Public License v3](LICENSE).