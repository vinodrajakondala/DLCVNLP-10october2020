# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'imgdispaly.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(510, 384)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame1 = QtWidgets.QFrame(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.frame1.setFont(font)
        self.frame1.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame1.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame1.setObjectName("frame1")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.frame1)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.frame_2 = QtWidgets.QFrame(self.frame1)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.frame_2)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_2 = QtWidgets.QLabel(self.frame_2)
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_2.setScaledContents(True)
        self.label_2.setAlignment(QtCore.Qt.AlignJustify|QtCore.Qt.AlignVCenter)
        self.label_2.setWordWrap(True)
        self.label_2.setIndent(1)
        self.label_2.setObjectName("label_2")
        self.gridLayout_3.addWidget(self.label_2, 0, 0, 1, 2)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.lineEdit_1 = QtWidgets.QLineEdit(self.frame_2)
        self.lineEdit_1.setObjectName("lineEdit_1")
        self.gridLayout_2.addWidget(self.lineEdit_1, 0, 1, 1, 1)
        self.Image_Path = QtWidgets.QLabel(self.frame_2)
        self.Image_Path.setObjectName("Image_Path")
        self.gridLayout_2.addWidget(self.Image_Path, 0, 0, 1, 1)
        self.toolButton_uploadImage = QtWidgets.QToolButton(self.frame_2)
        self.toolButton_uploadImage.setObjectName("toolButton_uploadImage")
        self.gridLayout_2.addWidget(self.toolButton_uploadImage, 0, 2, 1, 1)
        self.gridLayout_3.addLayout(self.gridLayout_2, 1, 0, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.saveImage = QtWidgets.QPushButton(self.frame_2)
        self.saveImage.setObjectName("saveImage")
        self.horizontalLayout.addWidget(self.saveImage)
        self.predict = QtWidgets.QPushButton(self.frame_2)
        self.predict.setObjectName("predict")
        self.horizontalLayout.addWidget(self.predict)
        self.gridLayout_3.addLayout(self.horizontalLayout, 1, 1, 1, 1)
        self.gridLayout_5.addWidget(self.frame_2, 0, 0, 1, 1)
        self.frame_4 = QtWidgets.QFrame(self.frame1)
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.gridLayout = QtWidgets.QGridLayout(self.frame_4)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(self.frame_4)
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.gridLayout_5.addWidget(self.frame_4, 1, 0, 1, 1)
        self.verticalLayout.addWidget(self.frame1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 510, 18))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_2.setText(_translate("MainWindow", "iNeuron Warehouse Equipment Detection using Detectron2 "))
        self.lineEdit_1.setPlaceholderText(_translate("MainWindow", "image path..."))
        self.Image_Path.setText(_translate("MainWindow", "Image Path"))
        self.toolButton_uploadImage.setText(_translate("MainWindow", "..."))
        self.saveImage.setText(_translate("MainWindow", "SaveImage"))
        self.predict.setText(_translate("MainWindow", "Predict"))
        self.label.setText(_translate("MainWindow", "TextLabel"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
