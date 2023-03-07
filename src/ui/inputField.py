import os
import torch
import backend
from PySide6.QtWidgets import QWidget, QPushButton, QVBoxLayout, \
    QTreeWidget, QTreeWidgetItem, QCheckBox
from PySide6.QtCore import Qt

class ListWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.links = []
        self.entries = []
        self.ConvNet = backend.ConvNetD()
        self.ConvNet.load_state_dict(torch.load("./D1.pt"))
        
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
        if len(self.entries) == 0: return
        
        for i in self.entries:
            pred = backend.predictMusic(i.text(0), self.ConvNet)
            print(f"{i.text(1)} - {pred}")

        #self.treeWidget.clear()
        #self.entries.clear()
        

    def handleItemClick(self):
        item = self.treeWidget.currentItem()
        backend.displaySpectrogram(item.text(0), item.text(1))

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

        
        