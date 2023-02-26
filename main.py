import spectrogram as sg

generator = sg.SpectrogramGenerator("./samples/sample.wav")
generator.saveSpectrogram("./output/output.png")
