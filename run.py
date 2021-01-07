import sys
from PyQt5.QtWidgets import *
from win_main import MyWin


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWin()
    sys.exit(app.exec_())