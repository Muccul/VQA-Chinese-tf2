from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import config as cfg
import cv2

class TabMain(QTabWidget):

    def __init__(self, parent=None):
        super(TabMain, self).__init__(parent)

        self.resize(1080, 600)
        # 创建5个选项卡小控件窗口
        self.tab_use = QWidget()
        self.tab_attention = QWidget()

        # 将2个选项卡添加到顶层窗口中
        self.addTab(self.tab_use, "VQA使用")
        self.addTab(self.tab_attention, "注意力可视化")

        self.setFont(QFont("Microsoft YaHei", 10))

        # 选项卡自定义的内容
        self.tab_useUI()
        self.tab_attentionUI()


    def tab_useUI(self):
        self.setTabText(0, "VQA使用")

        self.main_layout = QHBoxLayout()
        self.tab_use.setLayout(self.main_layout)


    def tab_attentionUI(self):
        self.setTabText(1, "注意力可视化")

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)

        self.figure_hbox = QHBoxLayout()
        self.figure_hbox.addWidget(self.canvas)

        self.tab_attention.setLayout(self.figure_hbox)


    def visual(self):
        plt.clf()
        att_value = cfg.att[1][0]
        att_value = att_value.numpy()
        att_value = att_value.reshape((14, 14, 1))
        plt.subplot(231)
        plt.imshow(att_value)

        heatmap = att_value / np.max(att_value)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
        plt.subplot(232)
        plt.imshow(heatmap)

        # img_ori = cv2.imread(cfg.img_path)
        img_ori = cv2.imdecode(np.fromfile(cfg.img_path,dtype=np.uint8),-1)
        img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
        heatmap = cv2.resize(heatmap, img_ori.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
        plt.subplot(233)
        plt.imshow(heatmap)

        heatmap_aug = np.uint8(np.clip((2 * (np.int16(heatmap) - 60) + 50), 0, 255))
        plt.subplot(234)
        plt.imshow(heatmap_aug)

        superimposed_img = (heatmap_aug * .6 + img_ori * 0.4).astype(np.uint8)
        plt.subplot(235)
        plt.imshow(img_ori)
        plt.subplot(236)
        plt.imshow(superimposed_img)

        self.canvas.draw()



