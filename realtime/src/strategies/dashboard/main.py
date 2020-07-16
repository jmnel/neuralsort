import sys
import os

from PySide2.QtWidgets import QApplication

from main_window import MainWindow


def main():

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
