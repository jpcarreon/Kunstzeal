import spectrogram as sg


sg.displaySpectrogram("./samples/sample.wav", size_x=10, size_y=6)
sg.saveSpectrogram("./samples/sample.wav", "./output/output.png")

