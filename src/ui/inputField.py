from PySide6.QtWidgets import QWidget, QPushButton, QVBoxLayout, QSplitter, \
    QTreeWidget, QTreeWidgetItem, QMenu
from PySide6.QtCore import Qt, QEvent
from PySide6.QtGui import QCursor, QColor
import os
import torch
import backend
import threads


class ListWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.links = []
        self.inputEntries = []
        self.currentCursor = QCursor()

        # loads CNN model and pretrained model; Model must correspond to the correct .pt file and vice-versa
        self.ConvNet = backend.ConvNetD()
        self.ConvNet.load_state_dict(torch.load("./D1.pt", torch.device("cpu")))
        
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
        self.outputTreeWidget.setHeaderLabels(["path", "Filename", "Format", "Label", "Prediction", "Mismatch?"])
        self.outputTreeWidget.hideColumn(0)
        self.outputTreeWidget.installEventFilter(self)
        self.outputTreeWidget.setColumnWidth(1, 300)
        self.outputTreeWidget.setColumnWidth(2, 10)
        self.outputTreeWidget.setColumnWidth(3, 80)
        self.outputTreeWidget.setColumnWidth(4, 80)

        innerLayout.addWidget(self.inputTreeWidget)
        innerLayout.addWidget(self.outputTreeWidget)
        mainLayout.addWidget(innerLayout)
        mainLayout.addWidget(self.runButton)

        innerLayout.setStretchFactor(1, 1)
        innerLayout.setStretchFactor(2, 3)

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
        """
            Retrieves signal that an audio file has finished processing.
            Moves the file from the input list to the output list and displays relevant data.
        """

        fileName = data["fp"].split("/")[-1]
        fileType = fileName.split(".")[-1]
        print(f"{fileName:10s} - {data['pred']}")

        newEntry = QTreeWidgetItem(self.outputTreeWidget, [
            data["fp"], 
            fileName, 
            fileType,
            data["label"], 
            data["pred"],
            str(data["mismatch"])
        ])
        if data["mismatch"]:
            newEntry.setBackground(5, QColor(255, 0, 0, 127))

        # remove entry in the input list
        self.inputTreeWidget.takeTopLevelItem(0)
        self.inputEntries.pop(0)

    def handleViewSpectrogram(self, item):
        # ensure viewing is allowed and the program is not busy
        print(f"{item.text(0)} - {item.isDisabled()}")
        if item.isDisabled() or self.currentCursor.shape() != Qt.ArrowCursor: return

        backend.displaySpectrogram(item.text(0), item.text(1))

    def handleDeleteItem(self, source, item):
        # removes item from the list
        source.takeTopLevelItem(source.indexFromItem(item).row())
        
        self.links.pop(self.links.index(item.text(0)))
        if item in self.inputEntries: 
            self.inputEntries.pop(self.inputEntries.index(item))
        
    def runPredictions(self):
        """
            Creates a thread that will process all audio files in the input list.
            Populates the output list once predictions are made.
        """

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
        """
            Checks if dragged files contain the accepted file format: mp3 or flac
        """
        for fp in event.mimeData().urls():
            fp = fp.toString()
            
            if fp[-3:] != "mp3" and fp[-4:] != "flac":
                event.ignore()
                return False
            
            # ignore internet links
            elif fp[8:] == "https://" or fp[7:] == "http://":
                event.ignore()
                return False
        
        return True

    def eventFilter(self, source, event):
        """
            Detects right clicks in the QTreeWidget items
        """

        if event.type() == QEvent.ContextMenu and \
            (source is self.inputTreeWidget or source is self.outputTreeWidget):

            # check if an item is correctly highlighted
            item = source.selectedItems()
            if len(item) == 0: return False

            contextMenu = QMenu()
            contextMenu.addAction("View Spectrogram").triggered.connect(
                lambda _: self.handleViewSpectrogram(item[0]))
            contextMenu.addAction("Remove").triggered.connect(
                lambda _: self.handleDeleteItem(source, item[0]))
            
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
        """
            Event where the file is dropped into the program
        """

        if not self.checkFileFormat(event): 
            return event.ignore()

        for i in event.mimeData().urls():
            i = i.toLocalFile()

            # ignore duplicate files
            if i in self.links: continue
            else: self.links.append(i)

            fileName = i.split("/")[-1]

            newEntry = QTreeWidgetItem(self.inputTreeWidget, [i, fileName])

            self.inputEntries.append(newEntry)

        if len(self.inputEntries) > 0 and self.currentCursor.shape() == Qt.ArrowCursor:
            self.runButton.setDisabled(False)
        
    def addFiles(self, files):
        for i in files:
            if i in self.links: continue
            else: self.links.append(i)

            fileName = i.split("/")[-1]

            newEntry = QTreeWidgetItem(self.inputTreeWidget, [i, fileName])

            self.inputEntries.append(newEntry)
        
        if len(self.inputEntries) > 0 and self.currentCursor.shape() == Qt.ArrowCursor:
            self.runButton.setDisabled(False)