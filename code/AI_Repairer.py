import sys
import cv2

from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import *
from repairer import Repair


class Ui_MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)

        self.image = None
        self.image_show = None
        self.image_repair = None
        self.image_show_repair = None

        self.repairer = Repair()
        self.set_ui()
        self.slot_init()

    def set_ui(self):

        self.__layout_main = QtWidgets.QHBoxLayout()
        self.__layout_fun_button = QtWidgets.QVBoxLayout()
        self.__layout_data_show = QtWidgets.QVBoxLayout()

        self.button_open_file = QtWidgets.QPushButton(u'打开文件')
        self.button_repair = QtWidgets.QPushButton(u'修复')
        self.button_switch = QtWidgets.QPushButton(u'原图')
        self.button_save = QtWidgets.QPushButton(u'保存')
        self.button_close = QtWidgets.QPushButton(u'退出')

        # Button color & height
        for button in [self.button_open_file, self.button_repair, self.button_switch, self.button_save,
                       self.button_close]:
            button.setStyleSheet("QPushButton{color:black}"
                                 "QPushButton:hover{color:white}"
                                 "QPushButton{background-color:rgb(26,148,230)}"
                                 "QPushButton{border:2px}"
                                 "QPushButton{border-radius:10px}"
                                 "QPushButton{padding:2px 4px}")

            button.setMinimumHeight(50)

        # move
        self.move(250, 100)

        # show
        #self.label_show_image = QtWidgets.QLabel()
        self.label_move = QtWidgets.QLabel()
        self.label_move.setFixedSize(100, 100)

        #self.label_show_image.setFixedSize(1280, 720)
        #self.label_show_image.setAutoFillBackground(False)

        self.__layout_fun_button.addWidget(self.button_open_file)
        self.__layout_fun_button.addWidget(self.button_repair)
        self.__layout_fun_button.addWidget(self.button_switch)
        self.__layout_fun_button.addWidget(self.button_save)
        self.__layout_fun_button.addWidget(self.button_close)
        self.__layout_fun_button.addWidget(self.label_move)

        self.__layout_main.addLayout(self.__layout_fun_button)
        #self.__layout_main.addWidget(self.label_show_image)

        # 创建带滚动条的label显示图片
        self.topFiller = QWidget()
        self.topFiller.setMinimumSize(1280, 720)  #设置滚动界面的尺寸
        self.label_show_image = QLabel(self.topFiller)
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.topFiller)
        self.scroll.setFixedSize(1280, 720)
        self.hSb = self.scroll.verticalScrollBar()
        vbox1 = QVBoxLayout()
        vbox1.addWidget(self.label_show_image)
        self.topFiller.setLayout(vbox1)
        vbox2 = QVBoxLayout()
        vbox2.addWidget(self.topFiller)
        self.scroll.setLayout(vbox2)
        self.__layout_main.addWidget(self.scroll)

        self.setLayout(self.__layout_main)
        self.label_move.raise_()
        self.setWindowTitle(u'AI 修复人像')

    def slot_init(self):
        self.button_open_file.clicked.connect(self.open_file)
        self.button_repair.clicked.connect(self.repair)
        self.button_switch.clicked.connect(self.image_switch)
        self.button_save.clicked.connect(self.save)
        self.button_close.clicked.connect(self.close)

    def wheelEvent(self, event):
        angle = event.angleDelta()
        y = angle.y()
        self.hSb.setValue(self.hSb.value() - y)

    def open_file(self):
        file = QFileDialog.getOpenFileName(self, '选择图像', filter='*.png;*.jpg')
        if not file[0] == '':
            print(file[0])
            self.image = cv2.imread(file[0])
            self.image_show = self.image
            height, width = self.image.shape[:2]
            self.image_show = cv2.resize(self.image, (width * 4, height * 4), interpolation=cv2.INTER_NEAREST)
            #self.image_show = cv2.resize(self.image, (1280, 720), interpolation=cv2.INTER_NEAREST)
            self.show_image(self.image_show)

    def show_image(self, show):
        # 设置滚动界面的尺寸
        height, width = show.shape[:2]
        self.topFiller.setMinimumSize(width, height)
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        show_image = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        self.label_show_image.setPixmap(QtGui.QPixmap.fromImage(show_image))

    def save(self):
        if self.image_repair is not None:
            file = QFileDialog.getSaveFileName(self, '保存图像', filter='*.png')
            if not file[0] == '':
                cv2.imwrite(file[0], self.image_repair)

    def repair(self):
        if self.image is not None:
            self.image_repair = self.repairer.repair(self.image)
            self.image_show_repair = self.image_repair
            #self.image_show_repair = cv2.resize(self.image_repair, (1280, 720), interpolation=cv2.INTER_NEAREST)
            self.show_image(self.image_show_repair)

    def image_switch(self):
        if self.button_switch.text() == '原图' and self.image_show is not None:
            self.button_switch.setText('修复图')
            self.show_image(self.image_show)
        elif self.button_switch.text() == '修复图' and self.image_show_repair is not None:
            self.button_switch.setText('原图')
            self.show_image(self.image_show_repair)

    def closeEvent(self, event):
        ok = QtWidgets.QPushButton()
        cacel = QtWidgets.QPushButton()

        msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u"关闭", u"是否关闭！")

        msg.addButton(ok, QtWidgets.QMessageBox.ActionRole)
        msg.addButton(cacel, QtWidgets.QMessageBox.RejectRole)
        ok.setText(u'确定')
        cacel.setText(u'取消')
        if msg.exec_() == QtWidgets.QMessageBox.RejectRole:
            event.ignore()
        else:
            event.accept()


if __name__ == "__main__":
    App = QApplication(sys.argv)
    ex = Ui_MainWindow()
    ex.show()
    sys.exit(App.exec_())
