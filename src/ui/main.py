from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog
from PySide6.QtGui import QIcon
import sys
import os
import inputField

class MainWindow(QMainWindow):
    def __init__(self, version):
        super().__init__()
        self.app = app
        self.version = version
        self.setWindowTitle(f"Kunstzeal v{version}")
        self.setWindowIcon(QIcon("icon.ico"))
        self.menuBarSetup()

        self.setMinimumSize(1000, 600)

        self.setCentralWidget(inputField.ListWidget())

    def menuBarSetup(self):
        """
            Construct menu bar
        """

        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        help_menu = menu_bar.addMenu("Help")

        file_menu.addAction("Open").triggered.connect(self.openFileDialog)
        file_menu.addSeparator()
        file_menu.addAction("Quit").triggered.connect(self.app.quit)
        
        help_menu.addAction("About").triggered.connect(self.displayAboutInfo)
    
    def displayAboutInfo(self):
        """
            Construct message about when Help->About is clicked
        """

        aboutBox = QMessageBox()
        aboutBox.setIcon(QMessageBox.Information)
        aboutBox.setWindowTitle("About")
        aboutBox.setText(f"Kunstzeal v{self.version}\
            \n\nKunstzeal is a program used to aid users in spectral/spectrogram analysis. Spectral analysis is commonly employed to detect improperly transcoded music files.\
            \n\nKunstzeal uses a pretrained CNN model to classify music files you input. Currently, the model is correct 97% of the time.\
            \n\nRepo: https://github.com/jpcarreon/Kunstzeal \
            \nThis program is distributed for free under the terms of the GNU General Public License v3"
        )

        aboutBox.exec()
    
    def openFileDialog(self):
        """
            Function to initiate file picker
        """
        files = QFileDialog.getOpenFileNames(self, 
                "Open music file/s", os.getcwd(), "Audio Files (*.mp3 *.flac)")
        self.centralWidget().addFiles(files[0])

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow("1.0")
    window.show()
    app.exec()