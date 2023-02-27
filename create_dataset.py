import os
import spectrogram as sg

count = 1
inputPath = ".\input"

print("Generating Spectrogram Data Set...\n")

for folder in os.listdir(inputPath):
    for fp in os.listdir(os.path.join(inputPath, folder)):
        print(f"Current: {folder}, {fp}")

        currentFile = os.path.join(inputPath, folder, fp)
        savePath = os.path.join(".\data_output", folder, fp[:-4]) + ".png"
        sg.saveSpectrogram(currentFile, savePath, size_x = 2.56, size_y = 2.56)