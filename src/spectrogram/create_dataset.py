import os
import spectrogram as sg

print("Generating Spectrogram Data Set...\n")

classification = {
    "FLAC": 0, "V0": 1, "320K": 2,
    "192K": 3, "128K": 4
}
inputPath = "./input"
outputPath = "./output"

if not os.path.exists(os.path.join(outputPath, "spectrograms")):
    os.makedirs(os.path.join(outputPath, "spectrograms"))

dataset_csv = open(f"{outputPath}/spectral_record.csv", "w")

for folder in os.listdir(inputPath):
    counter = 0
    filesNum = len(os.listdir(os.path.join(inputPath, folder)))
    for fp in os.listdir(os.path.join(inputPath, folder)):
        print(f"Current: [{folder:4s}] {fp}", end=" ")

        currentFile = os.path.join(inputPath, folder, fp)
        savePath = os.path.join(outputPath, "spectrograms", f"{folder}_{counter}.png")

        sg.saveSpectrogram(currentFile, savePath, size_x=2.56, size_y=2.56)
        dataset_csv.write(f"{folder}_{counter}.png, {classification[folder]}\n")

        counter += 1
        print(f"{counter}/{filesNum}")

dataset_csv.close()
