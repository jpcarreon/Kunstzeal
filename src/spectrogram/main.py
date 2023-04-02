from time import time
import spectrogram as sg
import os



#sg.displaySpectrogram("./oratorio.mp3", size_x=10, size_y=6)
#sg.saveSpectrogram("sample.mp3", "./output/output.png", size_x=2.56, size_y=2.56)
timings = {}
for idx, i in enumerate(os.listdir("./input/Samples")):
    start = time()
    sg.saveSpectrogram(f"./input/Samples/{i}", f"./output/{idx}.png")

    timings[i] = round(time() - start, 4)

print("\nOverall Results:")
for i, j in timings.items():
    print(f"{i}: {j}")