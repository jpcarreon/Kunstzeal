import backend
from mutagen.mp3 import MP3
from PySide6.QtCore import QThread, Signal

class predictionWorker(QThread):
    progress = Signal(dict)
    finished = Signal(int)

    def run(self):
        try:
            for fp in self.entries:
                pred = backend.predictMusic(fp, self.net)

                label, mismatch = self.checkMismatch(fp, pred)
                
                self.progress.emit({
                    "fp": fp,
                    "pred": pred,
                    "label": label,
                    "mismatch": mismatch
                })
        except:
            self.finished.emit(0)
        else:
            self.finished.emit(1)
    
    def checkMismatch(self, fp, pred):
        fileType = fp.split(".")[-1]

        if fileType == "mp3":
            audioData = MP3(fp)
            bitrateMode = str(audioData.info.bitrate_mode)[-3:]
            
            if bitrateMode == "VBR":
                label = "V0 VBR"
                mismatch = pred != "V0"
            else:
                label = f"{audioData.info.bitrate // 1000}K CBR"
                mismatch = pred != f"{audioData.info.bitrate // 1000}K"
        else:
            label = "FLAC"
            mismatch = pred != "FLAC"
        
        return (label, mismatch)

class spectrogramWorker(QThread):
    finished = Signal(int)

    def run(self):
        try:
            backend.displaySpectrogram(self.fp, self.name)
        except:
            self.finished.emit(0)
        else:
            self.finished.emit(1)