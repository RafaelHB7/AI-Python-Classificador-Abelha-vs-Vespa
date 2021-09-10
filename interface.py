from PyQt5 import QtCore, QtGui, QtWidgets

import numpy

from keras.preprocessing import image
from keras.models import model_from_json

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        self.labelImage = QtWidgets.QLabel(self.centralwidget)
        self.labelImage.setGeometry(QtCore.QRect(50, 50, 300, 200))
        self.labelImage.setAutoFillBackground(False)
        self.labelImage.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.labelImage.setText("")
        self.labelImage.setObjectName("labelImage")
        
        self.pushButtonCarregar = QtWidgets.QPushButton(self.centralwidget)
        self.pushButtonCarregar.setGeometry(QtCore.QRect(135, 310, 130, 40))
        self.pushButtonCarregar.setObjectName("pushButtonCarregar")
        
        self.imagePath = None
        self.pushButtonCarregar.clicked.connect(self.abrirImagem)
        
        self.pushButtonVerificar = QtWidgets.QPushButton(self.centralwidget)
        self.pushButtonVerificar.setGeometry(QtCore.QRect(135, 510, 130, 40))
        self.pushButtonVerificar.setObjectName("pushButtonVerificar")
        
        self.labelResultado = QtWidgets.QLabel(self.centralwidget)
        self.labelResultado.setGeometry(QtCore.QRect(165, 430, 70, 30))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.labelResultado.setFont(font)
        self.labelResultado.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.labelResultado.setFrameShape(QtWidgets.QFrame.Box)
        self.labelResultado.setText("")
        self.labelResultado.setObjectName("labelResultado")
        
        self.pushButtonZerar = QtWidgets.QPushButton(self.centralwidget)
        self.pushButtonZerar.setGeometry(QtCore.QRect(100, 430, 30, 30))
        
        self.pushButtonZerar.clicked.connect(self.zerar)
        
        self.pushButtonCarregarRNA = QtWidgets.QPushButton(self.centralwidget)
        self.pushButtonCarregarRNA.setGeometry(QtCore.QRect(540, 90, 130, 40))
        self.pushButtonCarregarRNA.setObjectName("pushButtonCarregarRNA")
        
        self.rnaPath = None
        self.pushButtonCarregarRNA.clicked.connect(self.abrirRNA)
        
        self.pushButtonCarregarPesos = QtWidgets.QPushButton(self.centralwidget)
        self.pushButtonCarregarPesos.setGeometry(QtCore.QRect(540, 170, 130, 40))
        self.pushButtonCarregarPesos.setObjectName("pushButtonCarregarPesos")
        
        self.pesosPath = None
        self.pushButtonCarregarPesos.clicked.connect(self.abrirPesos)
        
        self.pushButtonVerificar.clicked.connect(self.verificar)
        
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Abelha ou Vespa"))
        self.pushButtonCarregar.setText(_translate("MainWindow", "Carregar imagem"))
        self.pushButtonVerificar.setText(_translate("MainWindow", "Verificar"))
        self.pushButtonCarregarRNA.setText(_translate("MainWindow", "Carregar RNA"))
        self.pushButtonCarregarPesos.setText(_translate("MainWindow", "Carregar Pesos"))

    def zerar(self):
        self.labelResultado.setText("")

    def abrirImagem(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
                        None,
                        "Selecione uma imagem",
                        "",
                        "All Files (*);;Python Files (*.py)",
                        options=options)
        if fileName:
            pixmap = QtGui.QPixmap(fileName)
            self.labelImage.setPixmap(pixmap)
            self.labelImage.setScaledContents(True)
            self.imagePath = fileName
        
    def abrirRNA(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
                        None,
                        "Selecione a RNA",
                        "",
                        "All Files (*);;Python Files (*.py)",
                        options=options)
        if fileName:
            self.rnaPath = fileName
            
    def abrirPesos(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
                        None,
                        "Selecione a o arquivo de Pesos",
                        "",
                        "All Files (*);;Python Files (*.py)",
                        options=options)
        if fileName:
            self.pesosPath = fileName
        
    def verificar(self):
        if self.rnaPath == None or self.pesosPath == None or self.imagePath == None:
            self.labelResultado.setText("Erro")
        else:
            arquivo = open(self.rnaPath, 'r')
            estrutura_rede = arquivo.read()
            arquivo.close()
            
            classificador = model_from_json(estrutura_rede)
            classificador.load_weights(self.pesosPath)
            
            imagem_teste = image.load_img(self.imagePath, target_size=(64,64))
            
            imagem_teste = image.img_to_array(imagem_teste)
            imagem_teste /= 255
            
            imagem_teste = numpy.expand_dims(imagem_teste, axis = 0)
            
            previsao = classificador.predict(imagem_teste)
            previsao = (previsao > 0.5)
            
            if previsao:
                self.labelResultado.setText("Vespa")
            else:
                self.labelResultado.setText("Abelha")

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
