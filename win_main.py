import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from GUI.label_drag_img import ImgDrag
import config as cfg
from utils import tools as utl
import time
import jieba
import tensorflow as tf
from GUI.tab_main import TabMain

class MyWin(QMainWindow):

    def __init__(self):
        super(MyWin, self).__init__()

        self.initUI()
        self.initData()

    def initUI(self):
        self.resize(1080, 720)
        self.setMinimumSize(1080, 720)
        self.setMaximumSize(1080, 720)
        self.center()

        self.statusBar()

        self.tab_main = TabMain(parent=self)
        self.tab_main.move(1, 0)

        left = QFrame(self)
        left.setFrameShape(QFrame.StyledPanel)
        left.resize(600, 600)
        self.img_drag = ImgDrag(left)
        self.img_drag.setGeometry(80, 50, 480, 480)

        right = QFrame(self)
        right.setFrameShape(QFrame.StyledPanel)
        right.resize(400, 600)

        self.question_text = QTextEdit(right)
        self.question_text.setGeometry(20, 50, 280, 60)
        self.question_text.setPlaceholderText("请输入问题")

        self.question_button = QPushButton("回答", parent=right)
        self.question_button.setGeometry(320, 70, 80, 30)
        self.question_button.clicked.connect(self.click_vqa)

        self.answer_text = QTextEdit(right)
        self.answer_text.setReadOnly(True)
        self.answer_text.setGeometry(20, 135, 380, 70)

        self.main_tips = QTextEdit(right)
        self.main_tips.setFont(QFont("Microsoft YaHei", 12))
        self.main_tips.setGeometry(20, 240, 380, 300)
        self.main_tips.setEnabled(False)
        self.main_tips.setText("使用说明"
                               "\n系统：中文视觉问答VQA-Version1.0"
                               "\n功能：对一张图像提出问题，推理出可能答案"
                               "\n导航栏：Tab栏包括主页面和可视化界面"
                               "\n图片选择：点击选择按钮或者拖入候选框"
                               "\n问题输入：在问题框内输入待推理问题"
                               "\n推理答案：点击回答按钮即可进行答案推理"
                               "\n答案显示：答案框内将显示推理出的答案"
                               "\n日志栏：记录用户操作,并打印在日志栏中"
                               "\n可视化：展示推理过程中注意力机制的可视化结果"
                               "\n说明：参照使用说明，使用该系统")
        self.main_tips.setAlignment(Qt.AlignCenter)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)


        self.tab_main.main_layout.addWidget(splitter)
        # self.main_layout.addWidget(splitter)
        # self.use_main.setLayout(self.main_layout)

        self.log_main = QTextEdit(self)
        self.log_main.resize(1076, 100)
        self.log_main.setReadOnly(True)
        self.log_main.move(2, 600)

        self.setWindowTitle("中文视觉问答VQA")
        self.setWindowIcon(QIcon('GUI/icon.png'))
        self.show()

    def initData(self):
        self.printLog('Loading Model is OK！')

    def center(self):
        # 获得窗口
        qr = self.frameGeometry()
        # 获得屏幕中心点
        cp = QDesktopWidget().availableGeometry().center()
        # 显示到屏幕中心
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def printLog(self, text_log):
        self.log_main.insertHtml("  <b>User: </b>" + str(text_log) + "<br>")
        self.log_main.moveCursor(QTextCursor.End)


    def click_vqa(self):
        cfg.question_text = self.question_text.toPlainText() + '？'
        question_list = ['<boa>'] + list(jieba.cut(utl.cat_to_chs(cfg.question_text))) + ['<eoa>']
        cfg.question_seq = []
        for i in question_list:
            if i in cfg.tokenizer_que.word_index:
                cfg.question_seq.append(cfg.tokenizer_que.word_index[i])
            else:
                cfg.question_seq.append(cfg.tokenizer_que.word_index['<unk>'])
        while len(cfg.question_seq) < 20:
            cfg.question_seq.append(0)
        cfg.question_seq = tf.reshape(cfg.question_seq, (1, -1))
        self.start_thread()

    def vqa_visual(self, use_time):
        self.tab_main.visual()
        self.printLog('推理耗时：%6fs' % float(use_time))
        self.printLog(cfg.question_text + '答案是：' + cfg.answer)
        self.answer_text.setText(cfg.answer)

    def start_thread(self):
        self.Thread = Thread_VQA()
        self.Thread.signal.connect(self.vqa_visual)
        self.printLog('答案生成中...')
        self.Thread.start()



def vqa():
    st = time.time()
    img = utl.get_img_tensor(cfg.img_path)
    img_features = cfg.vgg16_extractor(img)
    img_features = tf.reshape(img_features, (1, -1, 512))
    out, att = cfg.model(cfg.question_seq, trg=None, image_info=img_features)
    cfg.answer = utl.convert_text(cfg.tokenizer_ans, out[0]).replace(' ', '')
    cfg.att = att
    end = time.time()
    return end-st


class Thread_VQA(QThread):
    signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()

    def __del__(self):
        self.wait()

    def run(self):
        # 进行任务操作
        time_process = vqa()
        self.signal.emit(str(time_process))



if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWin()
    sys.exit(app.exec_())