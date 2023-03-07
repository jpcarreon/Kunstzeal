from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox
import sys
import inputField

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.app = app
        self.setWindowTitle("Kunstzeal")
        self.menuBarSetup()

        self.setMinimumSize(1000, 600)
        self.setCentralWidget(inputField.ListWidget())

    def menuBarSetup(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        edit_menu = menu_bar.addMenu("Edit")
        help_menu = menu_bar.addMenu("Help")

        file_menu.addAction("Open")
        file_menu.addSeparator()
        file_menu.addAction("Quit").triggered.connect(self.app.quit)

        edit_menu.addAction("Save Selected")
        edit_menu.addAction("Remove Selected")
        
        help_menu.addAction("About").triggered.connect(self.displayAboutInfo)
    
    def displayAboutInfo(self):
        mBox = QMessageBox()
        mBox.setIcon(QMessageBox.Information)
        mBox.setMinimumSize(800, 500)
        mBox.setWindowTitle("About")
        mBox.setText("Hello!")
        mBox.setInformativeText("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse tortor augue, tincidunt vitae purus eu, fringilla congue mi. Cras vehicula sagittis accumsan. Aenean porta vitae urna ac pulvinar. Vivamus faucibus nunc a congue malesuada. Cras varius interdum libero in semper. Sed mauris felis, luctus nec enim at, ornare auctor ex. Donec suscipit cursus sem, vitae imperdiet metus imperdiet sed. Nam ultrices metus a lacus rutrum condimentum. Pellentesque commodo in justo at pulvinar.")

        mBox.exec()

    


app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()