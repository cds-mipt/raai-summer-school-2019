import sys
import os
import numpy as np
from PyQt5 import QtGui
from PyQt5.QtWidgets import QPushButton, QHBoxLayout, QVBoxLayout, QMainWindow
from PyQt5.QtWidgets import QRadioButton, QGroupBox, QFileDialog, QLabel, QSlider, QLineEdit, QProgressBar
from PyQt5.QtGui  import QPixmap, QImage, QPainter, QColor, QCursor
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtCore import Qt, QPoint, QObject, QThread, QTimer
import cv2
from ScenePainter import ScenePainter
from shapely.geometry import Polygon

import time

class CrossroadSimulatorGUI(QMainWindow):
    #Переопределяем конструктор класса:
    def __init__(self, master=None):
        QMainWindow.__init__(self, master)
        self.isMaskMode = False
        self.mode = 0
        self.isRecognize = False
        self.isRLMode = False
        self.timerPeriod = 35 #ms
        self.stepCounter = 0
        self.nCars = 4

        self.painter = ScenePainter()
        self.backgroundImage = self.painter.load_background()
        self.carLibrary = self.painter.load_cars_library()
        self.carSizes = self.painter.load_cars_sizes()
        self.paths = self.painter.load_expanded_car_paths(shift=100)
        self.bresenhamPaths = self.painter.get_bresenham_paths(self.paths)
        self.startTime = time.time()

        self.maskImage = None
        if self.backgroundImage is not None:
            self.maskImage = np.zeros((self.backgroundImage.shape[0],
                                       self.backgroundImage.shape[1]),dtype='uint8')

        self.currentImage = None
        self.cars = []

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.simulatorMotion)

        self.initUI()

    def initUI(self):

        self.mainwidget = QWidget(self)
        self.setCentralWidget(self.mainwidget)

        self.startButton = QPushButton("Start")
        self.stopButton = QPushButton("Stop")

        self.recognizeButton = QPushButton("Activate recognition")
        self.maskButton = QPushButton("Show mask")
        self.rlButton = QPushButton("Activate RL mode")

        self.stopButton.setEnabled(False)
        self.recognizeButton.setEnabled(False)
        self.maskButton.setEnabled(False)
        self.rlButton.setEnabled(False)

        self.carsNumber= QLabel('Number of cars:')
        self.leCarsNumber = QLineEdit('4')
        self.timeFromStart = QLabel('Step:')
        self.leTimeFromStart = QLineEdit('-')

        self.imageLabel = QLabel()

        self.defaultImWidth = 900
        self.defaultImHeight = 900


        self.imageLabel.setMinimumWidth(self.defaultImWidth)
        self.imageLabel.setMaximumWidth(self.defaultImWidth)
        self.imageLabel.setMinimumHeight(self.defaultImHeight)
        self.imageLabel.setMaximumHeight(self.defaultImHeight)

        self.startButton.clicked.connect(self.startButtonClicked)
        self.stopButton.clicked.connect(self.stopButtonClicked)
        self.recognizeButton.clicked.connect(self.recognizeButtonClicked)
        self.maskButton.clicked.connect(self.maskButtonClicked)
        self.rlButton.clicked.connect(self.rlButtonClicked)

        self.hbox = QHBoxLayout()

        self.hbox.addWidget(self.startButton)
        self.hbox.addWidget(self.stopButton)

        self.hbox.addWidget(self.recognizeButton)
        self.hbox.addWidget(self.maskButton)
        self.hbox.addWidget(self.rlButton)


        self.hbox.addWidget(self.carsNumber)
        self.hbox.addWidget(self.leCarsNumber)
        self.hbox.addWidget(self.timeFromStart)
        self.hbox.addWidget(self.leTimeFromStart)

        self.pixmap = QPixmap()

        self.mainhbox = QHBoxLayout()

        self.vbox = QVBoxLayout()
        self.vbox.addLayout(self.hbox)
        self.vbox.addWidget(self.imageLabel)

        self.setGeometry(0, 0, 900, 900)
        self.setWindowTitle('Crossroad Simulator v.1.0 - CDS Lab, MIPT')
        self.setWindowIcon(QtGui.QIcon('resources/CDS-lab.png'))

        self.mainhbox.addLayout(self.vbox)

        self.mainwidget.setLayout(self.mainhbox)


        self.currentImage = self.backgroundImage.copy()
        self.printImageOnLabel(self.backgroundImage, self.imageLabel)


    def printImageOnLabel(self, img, labelWidget):
        if (len(img.shape)==3):
            height, width, channel = img.shape
        else:
            height, width = img.shape
            channel = 1

        bytesPerLine = 3 * width
        if channel == 3:
            rgbCVimage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            rgbCVimage = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        qImg = QImage(rgbCVimage.data, width, height, bytesPerLine, QImage.Format_RGB888)

        self.pixmap = QPixmap(qImg)

        wP = self.pixmap.width()
        hP = self.pixmap.height()

        if (wP >= hP):
            labelNewWidth = self.defaultImWidth
            labelNewHeight = hP * self.defaultImWidth / wP
        else:
            labelNewWidth = wP * self.defaultImHeight / hP
            labelNewHeight = self.defaultImHeight

        labelWidget.setMinimumWidth(labelNewWidth)
        labelWidget.setMaximumWidth(labelNewWidth)
        labelWidget.setMinimumHeight(labelNewHeight)
        labelWidget.setMaximumHeight(labelNewHeight)

        # Вычисляем ширину окна изображения
        w = labelWidget.width()
        # Вычисляем высоту окна изображения
        h = labelWidget.height()
        self.pixmap = self.pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        labelWidget.setPixmap(self.pixmap)

        return True

    def getCarPolygon(self, car, longCoef = 1.0, rightCoef = 1.0):
        # polygon of car
        car_w, car_h = car['sizes']
        car_polygon = []
        # if longCoef is None:
        #     car_polygon.append([car['x'] - car_w / 2, car['y'] - car_h / 2, 1])
        #     car_polygon.append([car['x'] + car_w / 2, car['y'] - car_h / 2, 1])
        #     car_polygon.append([car['x'] + car_w / 2, car['y'] + car_h / 2, 1])
        #     car_polygon.append([car['x'] - car_w / 2, car['y'] + car_h / 2, 1])
        # elif rightCoef is None:
        #     car_polygon.append([car['x'] - car_w / 2, car['y'] - car_h / 2, 1])
        #     car_polygon.append([car['x'] + (car_w / 2)*longCoef, car['y'] - car_h / 2, 1])
        #     car_polygon.append([car['x'] + (car_w / 2)*longCoef, car['y'] + car_h / 2, 1])
        #     car_polygon.append([car['x'] - car_w / 2, car['y'] + car_h / 2, 1])
        # else:
        car_polygon.append([car['x'] - car_w / 2, car['y'] - car_h / 2, 1])
        car_polygon.append([car['x'] + (car_w / 2) * longCoef, car['y'] - car_h / 2, 1])
        car_polygon.append([car['x'] + (car_w / 2) * longCoef, car['y'] + (car_h / 2)*rightCoef, 1])
        car_polygon.append([car['x'] - car_w / 2, car['y'] + (car_h / 2)*rightCoef, 1])

        rotation_mat = cv2.getRotationMatrix2D((car['x'], car['y']), car['angle'], 1.0)
        # cosA = np.cos(np.radians(car['angle']))
        # sinA = np.sin(np.radians(car['angle']))
        # rotation_matrix = np.asarray([[cosA, sinA, car['x']],
        #                              [-sinA, cosA, car['y']]])

        car_polygon = np.transpose(np.dot(rotation_mat, np.transpose(np.asarray(car_polygon))))
        return car_polygon

    # def existTwoCarsIntersection(self, car, otherCar):
    #     carPlygon = self.getCarPolygon(car)
    #     otherCarPlygon = self.getCarPolygon(otherCar)
    #
    #     # Example polygon
    #     # xy = [[130.21001, 27.200001], [129.52, 27.34], [129.45, 27.1], [130.13, 26.950001]]
    #     carPlygonShapely = Polygon(carPlygon)
    #     otherCarPlygonShapely = Polygon(otherCarPlygon)
    #     # Example grid cell
    #     #gridcell_shape = box(129.5, -27.0, 129.75, 27.25)
    #     # The intersection
    #     intersectionArea = carPlygonShapely.intersection(otherCarPlygonShapely).area
    #     if intersectionArea > 0:
    #         return True
    #     else:
    #         return False

    def existTwoCarsIntersection(self, car, otherCar, longCoef = 1.0, rightCoef = 1.0):
        carPlygon = self.getCarPolygon(car, longCoef = longCoef, rightCoef = rightCoef)
        otherCarPlygon = self.getCarPolygon(otherCar)

        carPlygonShapely = Polygon(carPlygon)
        otherCarPlygonShapely = Polygon(otherCarPlygon)

        intersectionArea = carPlygonShapely.intersection(otherCarPlygonShapely).area
        if intersectionArea > 0:
            return True
        else:
            return False

    def existIntersections(self, car, cars, longCoef = 1.0, rightCoef = 1.0):
        isIntersection = False
        for otherCar in cars:
            if car['index'] != otherCar['index']:
                if self.existTwoCarsIntersection(car, otherCar, longCoef = longCoef, rightCoef = rightCoef):
                    isIntersection = True
                    break
        return isIntersection

    # def existIntersections(self, car, cars):
    #     isIntersection = False
    #     for otherCar in cars:
    #         if car['index'] != otherCar['index']:
    #             if self.existTwoCarsIntersection(car, otherCar):
    #                 isIntersection = True
    #                 break
    #     return isIntersection

    def initScene(self, nCars):

        if nCars > 8:
            nCars = 8
        self.cars = []
        self.currentImage = self.backgroundImage.copy()

        for i in range(nCars):
            isIntersection = True
            while(isIntersection):
                routeID = np.random.randint(0, len(self.bresenhamPaths))
                car = {}
                # car['x'] = self.currentImage.shape[1] / 2 + 40
                # car['y'] = self.currentImage.shape[1] / 2 + i * 150
                # car['angle'] = 90
                # car['speed'] = 3 #pixels per step
                if(i < 4):
                    car['counter'] = 0
                    car['x'] = self.bresenhamPaths[routeID][0][0]
                    car['y'] = self.bresenhamPaths[routeID][1][0]
                    car['angle'] = self.bresenhamPaths[routeID][2][0]
                else:
                    car['counter'] = 200 + np.random.randint(0, 150)
                    car['x'] = self.bresenhamPaths[routeID][0][car['counter']]
                    car['y'] = self.bresenhamPaths[routeID][1][car['counter']]
                    car['angle'] = self.bresenhamPaths[routeID][2][car['counter']]

                car['speed'] = np.random.randint(4, 10)  # pixels per step
                car['route'] = routeID
                car['image_index'] = np.random.randint(0, len(self.carLibrary))
                car['sizes'] = self.carSizes[car['image_index']] ##(width, height)
                car['index'] = i
                isIntersection = self.existIntersections(car, self.cars)
            self.cars.append(car)

            self.currentImage, self.maskImage = self.painter.show_car(x=car['x'], y=car['y'], angle=car['angle'],
                                                                      car_index=car['image_index'],
                                                                      background_image=self.currentImage,
                                                                      full_mask_image=self.maskImage)
        self.printImageOnLabel(self.currentImage, self.imageLabel)

    def simulatorMotion(self):
        if self.mode == 1:
            startTime = time.time()
            self.stepCounter += 1
            self.currentImage = self.backgroundImage.copy()
            # cv2.imwrite('data/cars/tmp/background_before.jpg', self.backgroundImage)
            self.maskImage = np.zeros((self.backgroundImage.shape[0],
                                       self.backgroundImage.shape[1]), dtype='uint8')
            newCars = []
            for car in self.cars:
                newCar = car.copy()

                longDistStep = newCar['counter'] + 6*newCar['speed']
                if (longDistStep < len(self.bresenhamPaths[car['route']][0])):
                    longDistAngle = self.bresenhamPaths[car['route']][2][longDistStep]
                else:
                    longDistAngle = newCar['angle']
                longDistAngleError = np.abs(np.abs(newCar['angle'])-np.abs(longDistAngle))
                if(longDistAngleError < 10):
                    longDistSpeed = int(newCar['speed']* (1 - longDistAngleError/90.0))
                else:
                    longDistSpeed = 3

                #newCar['counter'] = newCar['counter'] + newCar['speed']
                newCar['counter'] = newCar['counter'] + longDistSpeed
                if(newCar['counter'] < len(self.bresenhamPaths[car['route']][0])):
                    newCar['x'] = self.bresenhamPaths[car['route']][0][newCar['counter']]
                    newCar['y'] = self.bresenhamPaths[car['route']][1][newCar['counter']]
                    newCar['angle'] = self.bresenhamPaths[car['route']][2][newCar['counter']]
                    isIntersectionPrew = self.existIntersections(newCar, self.cars, longCoef=2.0, rightCoef=2.0)
                    isIntersectionNew = self.existIntersections(newCar, newCars, longCoef=2.0, rightCoef=2.0)
                    isIntersection = isIntersectionPrew or isIntersectionNew
                    if not isIntersection:
                        car = newCar

                else:
                    car['counter'] = 0
                    isIntersection = True
                    while (isIntersection):
                        routeID = np.random.randint(0, len(self.bresenhamPaths))
                        car['x'] = self.bresenhamPaths[routeID][0][0]
                        car['y'] = self.bresenhamPaths[routeID][1][0]
                        car['angle'] = self.bresenhamPaths[routeID][2][0]
                        car['speed'] = np.random.randint(4, 10)  # pixels per step
                        car['route'] = routeID
                        car['image_index'] = np.random.randint(0, len(self.carLibrary))
                        car['sizes'] = self.carSizes[car['image_index']]  ##(width, height)
                        car['counter'] = 0
                        isIntersection = self.existIntersections(car, self.cars)

                newCars.append(car)
                self.currentImage, self.maskImage = self.painter.show_car(x=car['x'], y=car['y'], angle=car['angle'],
                                                                          car_index=car['image_index'],
                                                                          background_image=self.currentImage,
                                                                          full_mask_image=self.maskImage)
            self.cars = newCars
            if(self.isMaskMode):
                self.printImageOnLabel(self.maskImage, self.imageLabel)
            else:
                self.printImageOnLabel(self.currentImage, self.imageLabel)
            endTime = time.time()
            stepPeriod = endTime - startTime

            self.leTimeFromStart.setText(str(self.stepCounter))
            #self.leTimeFromStart.setText("%.4f" % stepPeriod)
            # cv2.imwrite('data/cars/tmp/currentImage_after.jpg', self.currentImage)
            # cv2.imwrite('data/cars/tmp/background_after.jpg', self.backgroundImage)


    def startButtonClicked(self):

        if (self.mode == 0):
            self.startButton.setText('Pause')
            self.stopButton.setEnabled(True)
            self.recognizeButton.setEnabled(True)
            self.maskButton.setEnabled(True)
            self.rlButton.setEnabled(True)
            self.nCars = int(self.leCarsNumber.text())
            self.initScene(self.nCars)
            self.startTime = time.time()
            self.leTimeFromStart.setText('0')
            self.timer.start(self.timerPeriod)
            self.stepCounter = 0
            self.mode = 1
        elif self.mode==1:
            self.startButton.setText('Start')
            self.mode = 2
        elif self.mode == 2:
            self.startButton.setText('Pause')
            self.mode = 1

    def stopButtonClicked(self):
        self.startButton.setText('Start')
        self.stopButton.setEnabled(False)
        self.recognizeButton.setEnabled(False)
        self.maskButton.setEnabled(False)
        self.rlButton.setEnabled(False)
        self.timer.stop()
        self.nCars = int(self.leCarsNumber.text())
        self.initScene(self.nCars)
        self.mode = 0

    def recognizeButtonClicked(self):
        if (not self.isRecognize):
            self.recognizeButton.setText('Deactivate recognition')
            self.isRecognize = True
        else:
            self.recognizeButton.setText('Activate recognition')
            self.isRecognize = False

    def maskButtonClicked(self):

        if (not self.isMaskMode):
            self.printImageOnLabel(self.maskImage, self.imageLabel)

            self.maskButton.setText('Hide mask')
            self.isMaskMode = True
        else:
            self.printImageOnLabel(self.currentImage, self.imageLabel)

            self.maskButton.setText('Show mask')
            self.isMaskMode = False

    def rlButtonClicked(self):

        if (not self.isRLMode):
            self.rlButton.setText('Deactivate RL Mode')
            self.isRLMode = True
        else:
            self.rlButton.setText('Activate RL Mode')
            self.isRLMode = False

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = CrossroadSimulatorGUI()
    ex.move(0, 0)
    ex.show()

    sys.exit(app.exec_())