import os
import backend
from PySide6.QtWidgets import QWidget, QMainWindow, QPushButton, QVBoxLayout, \
    QTreeWidget, QTreeWidgetItem, QCheckBox
from PySide6.QtCore import Qt


"""
for (color, code) in colors:
    newItem = QTreeWidgetItem(treeWidget, [None, color, code])
    checkBox = QCheckBox()
    checkBox.stateChanged.connect(self.handleItemCheck)
    treeWidget.setItemWidget(newItem, 0, checkBox)
"""

class ListWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.links = []
        self.entries = []
        self.counter = 0
        self.ConvNet = backend.ConvNetD()
        
        layout = QVBoxLayout()
        
        runButton = QPushButton("Predict")
        runButton.clicked.connect(self.runPredictions)

        self.treeWidget = QTreeWidget()
        self.treeWidget.setHeaderLabels(["path", "Filename", "Format", "Prediction", "Actions"])
        self.treeWidget.hideColumn(0)
        self.treeWidget.itemDoubleClicked.connect(self.handleItemClick)


        layout.addWidget(self.treeWidget)
        layout.addWidget(runButton)
        self.setLayout(layout)
    
    def runPredictions(self):
        #backend.saveSpectrogram(i, f"./temp/spectral_{self.counter}.png")
        #self.counter += 1

        if not os.path.exists("./output"): os.makedirs("output")
        
        if len(self.entries) == 0: return
        
        for i in self.entries:
            savePath = f"./output/spectral_{self.counter}.png"
            self.counter += 1
            
            backend.saveSpectrogram(i.text(0), savePath)
            pred = self.ConvNet.predictSingle("./D1.pt", savePath)
            print(f"{i.text(1)} - {pred}")

        #self.treeWidget.clear()
        #self.entries.clear()
        

    def handleItemClick(self):
        item = self.treeWidget.currentItem()
        backend.displaySpectrogram(item.text(0))

    def checkFileFormat(self, event):
        for fp in event.mimeData().urls():
            fp = fp.toString()
            if fp[-3:] != "mp3" and fp[-4:] != "flac":
                event.ignore()
                return False
        
        return True
    
    def dragEnterEvent(self, event):
        if self.checkFileFormat(event): event.accept()
        else: event.ignore()

    def dragMoveEvent(self, event):
        if self.checkFileFormat(event): event.accept()
        else: event.ignore()

    def dropEvent(self, event):
        if not self.checkFileFormat(event): 
            return event.ignore()

        for i in event.mimeData().urls():
            i = i.toLocalFile()
            if i in self.links: continue

            self.links.append(i)
            fileName = i.split("/")[-1]
            fileType = fileName.split(".")[-1]
            self.entries.append(QTreeWidgetItem(
                self.treeWidget, [i, fileName, fileType, None, None]))

        
        