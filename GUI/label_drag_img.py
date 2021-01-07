import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import config

class ImgDrag(QLabel):
    """实现图片文件拖放功能"""

    def __init__(self, parent=None):
        super(ImgDrag, self).__init__(parent=parent)

        self.img_path = ''
        self.setAcceptDrops(True)  # 设置接受拖放动作
        self.pix = QPixmap('./GUI/img_drop.png')
        self.setPixmap(self.pix)
        self.setScaledContents(True)
        self.setCursor(QCursor(Qt.PointingHandCursor))

    def dragEnterEvent(self, e):
        file_path = e.mimeData().text()
        end_img = ('.png', '.jpg', '.jpeg', '.bmp')
        if file_path.endswith(end_img):  # 如果是.srt结尾的路径接受
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e): # 放下文件后的动作
        path = e.mimeData().text().replace('file:///', '')  # 删除多余开头
        config.img_path = path
        self.img_path = path
        self.pix = QPixmap(path)
        self.setPixmap(self.pix)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:  # 左键按下
            fname = QFileDialog.getOpenFileName(self, 'Open file', './', 'Images(*.jpg *.png *.jpeg *.bmp)')
            if fname[0]:
                config.img_path = fname[0]
                self.img_path = fname[0]
                self.pix = QPixmap(fname[0])
                self.setPixmap(self.pix)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ImgDrag()
    win.resize(200,200)
    print(win.size())
    win.show()
    sys.exit(app.exec_())

