"""
    Script used to generate spectrogram datasets.
    Requires a specific folder structure:
    root/ 
        - input/
            - FLAC/
            - V0/
            - 320K/
            - 192K/
            - 128K/    
        - output/
    
    The output will contain a spectrograms folder and spectral_record.csv
"""

import os

import sys; sys.path.append('../')
import spectrogram as sg
import spectrogram.utils as utils

print("Generating Spectrogram Data Set...\n")

classification = {
    "FLAC": 0, "V0": 1, "320K": 2,
    "192K": 3, "128K": 4
}
inputPath = "./input"
outputPath = "./output"

# Create spectrograms folder if it doesn't exist
os.makedirs(os.path.join(outputPath, "spectrograms"), exist_ok=True)


dataset_csv = open(f"{outputPath}/spectral_record.csv", "w")

for folder in os.listdir(inputPath):
    counter = 0
    filesNum = len(os.listdir(os.path.join(inputPath, folder)))
    for fp in os.listdir(os.path.join(inputPath, folder)):
        utils.displayProgress(counter, filesNum)

        currentFile = os.path.join(inputPath, folder, fp)
        savePath = os.path.join(outputPath, "spectrograms", f"{folder}_{counter}.png")

        # generate 256x256 spectrogram
        sg.saveSpectrogram(currentFile, savePath, size_x=2.56, size_y=2.56)
        dataset_csv.write(f"{folder}_{counter}.png, {classification[folder]}\n")

        counter += 1

dataset_csv.close()
