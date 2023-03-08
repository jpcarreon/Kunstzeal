import backend
from PySide6.QtCore import QThread, Signal

class predictionWorker(QThread):
    progress = Signal(dict)
    finished = Signal(int)

    def run(self):
        try:
            for fp in self.entries:
                #pred = backend.predictMusic(fp, self.net)
                pred=1
                
                self.progress.emit({
                    "fp": fp,
                    "pred": pred
                })
        except:
            self.finished.emit(0)
        else:
            self.finished.emit(1)

class spectrogramWorker(QThread):
    finished = Signal(int)

    def run(self):
        try:
            backend.displaySpectrogram(self.fp, self.name)
        except:
            self.finished.emit(0)
        else:
            self.finished.emit(1)