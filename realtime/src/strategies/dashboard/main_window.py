from PySide2.QtCore import Qt
from PySide2.QtWidgets import (QApplication, QMainWindow, QTableWidget, QTableWidgetItem)

import settings


class MainWindow(QMainWindow):

    def __init__(self, parent=None):

        super().__init__(parent)

        table = QTableWidget(12, 3, self)
        q = QTableWidgetItem(1)

        table.setItem(1, 1, q)
