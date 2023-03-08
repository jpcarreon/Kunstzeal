import os
import torch
import backend
import threads
from PySide6.QtWidgets import QWidget, QPushButton, QVBoxLayout, QSplitter, \
    QTreeWidget, QTreeWidgetItem, QMenu
from PySide6.QtCore import Qt, QEvent
from PySide6.QtGui import QCursor

class ListWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.links = []
        self.inputEntries = []
        self.currentCursor = QCursor()
        self.ConvNet = backend.ConvNetD()
        self.ConvNet.load_state_dict(torch.load("./D1.pt"))
        
        mainLayout = QVBoxLayout()
        innerLayout = QSplitter()
        
        self.runButton = QPushButton("Predict")
        self.runButton.setDisabled(True)
        self.runButton.clicked.connect(self.runPredictions)

        self.inputTreeWidget = QTreeWidget()
        self.inputTreeWidget.setHeaderLabels(["Input Files", "Filename"])
        self.inputTreeWidget.hideColumn(1)
        self.inputTreeWidget.installEventFilter(self)
        

        self.outputTreeWidget = QTreeWidget()
        self.outputTreeWidget.setHeaderLabels(["path", "Filename", "Format", "Prediction"])
        self.outputTreeWidget.hideColumn(0)
        self.outputTreeWidget.installEventFilter(self)

        innerLayout.addWidget(self.inputTreeWidget)
        innerLayout.addWidget(self.outputTreeWidget)
        mainLayout.addWidget(innerLayout)
        mainLayout.addWidget(self.runButton)
        self.setLayout(mainLayout)

    def changeCursor(self, cursorShape):
        self.currentCursor.setShape(cursorShape)
        self.setCursor(self.currentCursor)

    def handleThreadFinish(self, data):
        print(f"Thread finished: {data}")
        self.changeCursor(Qt.ArrowCursor)
        
        if len(self.inputEntries) > 0:
            self.runButton.setDisabled(False)
    
    def handleThreadProgress(self, data):
        fileName = data["fp"].split("/")[-1]
        fileType = fileName.split(".")[-1]
        print(f"{fileName:10s} - {data['pred']}")

        QTreeWidgetItem(self.outputTreeWidget, [
            data["fp"], 
            fileName, 
            fileType, 
            data["pred"]
        ])

        self.inputTreeWidget.takeTopLevelItem(0)
        self.inputEntries.pop(0)


    def runPredictions(self):
        self.runButton.setDisabled(True)
        worker = threads.predictionWorker(self)
        worker.entries = []
        worker.net = self.ConvNet

        worker.progress.connect(self.handleThreadProgress)
        worker.finished.connect(self.handleThreadFinish)
        worker.finished.connect(worker.deleteLater)

        for i in self.inputEntries:
            worker.entries.append(i.text(0))
            i.setDisabled(True)
        
        self.changeCursor(Qt.WaitCursor)
        worker.start()
        
    def checkFileFormat(self, event):
        for fp in event.mimeData().urls():
            fp = fp.toString()
            if fp[-3:] != "mp3" and fp[-4:] != "flac":
                event.ignore()
                return False
        
        return True
    
    def handleViewSpectrogram(self, item):
        print(f"{item.text(0)} - {item.isDisabled()}")
        if item.isDisabled() or self.currentCursor.shape() != Qt.ArrowCursor: return

        backend.displaySpectrogram(item.text(0), item.text(1))


    def eventFilter(self, source, event):
        if event.type() == QEvent.ContextMenu and \
            (source is self.inputTreeWidget or source is self.outputTreeWidget):

            item = source.selectedItems()
            if len(item) == 0: return False

            contextMenu = QMenu()
            contextMenu.addAction("View Spectrogram")
            contextMenu.addAction("Remove")

            contextMenu.triggered.connect(lambda _: self.handleViewSpectrogram(item[0]))
            
            contextMenu.exec(event.globalPos())

            return True

        return False

    def dragEnterEvent(self, event):
        # computationally heavy
        if self.checkFileFormat(event): event.accept()
        else: event.ignore()

    def dragMoveEvent(self, event):
        # computationally heavy
        if self.checkFileFormat(event): event.accept()
        else: event.ignore()

    def dropEvent(self, event):
        if not self.checkFileFormat(event): 
            return event.ignore()

        for i in event.mimeData().urls():
            i = i.toLocalFile()
            if i in self.links: continue
            else: self.links.append(i)

            fileName = i.split("/")[-1]

            newEntry = QTreeWidgetItem(self.inputTreeWidget, [i, fileName])

            self.inputEntries.append(newEntry)

        if len(self.inputEntries) > 0 and self.currentCursor.shape() == Qt.ArrowCursor:
            self.runButton.setDisabled(False)
        