from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog
from PySide6.QtGui import QIcon
import sys
import os
import inputField

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.app = app
        self.setWindowTitle("Kunstzeal")
        self.setWindowIcon(QIcon("icon.png"))
        self.menuBarSetup()

        self.setMinimumSize(1000, 600)

        self.setCentralWidget(inputField.ListWidget())

    def menuBarSetup(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        help_menu = menu_bar.addMenu("Help")

        file_menu.addAction("Open").triggered.connect(self.openFileDialog)
        file_menu.addSeparator()
        file_menu.addAction("Quit").triggered.connect(self.app.quit)
        
        help_menu.addAction("About").triggered.connect(self.displayAboutInfo)
    
    def displayAboutInfo(self):
        aboutBox = QMessageBox()
        aboutBox.setIcon(QMessageBox.Information)
        aboutBox.setMinimumSize(800, 500)
        aboutBox.setWindowTitle("About")
        aboutBox.setText("Hello!")
        aboutBox.setInformativeText("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse tortor augue, tincidunt vitae purus eu, fringilla congue mi. Cras vehicula sagittis accumsan. Aenean porta vitae urna ac pulvinar. Vivamus faucibus nunc a congue malesuada. Cras varius interdum libero in semper. Sed mauris felis, luctus nec enim at, ornare auctor ex. Donec suscipit cursus sem, vitae imperdiet metus imperdiet sed. Nam ultrices metus a lacus rutrum condimentum. Pellentesque commodo in justo at pulvinar.")

        aboutBox.exec()
    
    def openFileDialog(self):
        files = QFileDialog.getOpenFileNames(self, 
                "Open music file/s", os.getcwd(), "Audio Files (*.mp3 *.flac)")
        self.centralWidget().addFiles(files[0])

app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()