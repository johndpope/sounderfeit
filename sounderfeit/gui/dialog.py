#!/usr/bin/env python3

import numpy as np
from scipy.signal import lfilter, detrend, hamming

from PyQt5.QtCore import (pyqtProperty, pyqtSignal, QDataStream, QDateTime,
        QEvent, QEventTransition, QFile, QIODevice, QParallelAnimationGroup,
        QPointF, QPropertyAnimation, qrand, QRectF, QSignalTransition, qsrand,
        QState, QStateMachine, Qt, QTimer)
from PyQt5.QtGui import QColor, QPen, QPainter, QPainterPath, QPixmap
from PyQt5.QtWidgets import (QApplication, QGraphicsItem, QGraphicsObject,
        QGraphicsScene, QGraphicsTextItem, QGraphicsView)
from PyQt5.QtWidgets import (QDialog, QMenu, QMenuBar, QGroupBox,
                             QHBoxLayout, QPushButton, QGridLayout,
                             QLabel, QLineEdit, QTextEdit,
                             QFormLayout, QComboBox, QSpinBox,
                             QDialogButtonBox, QVBoxLayout, QSlider,
                             QRadioButton)


class Waveform(QGraphicsObject):
    def __init__(self, getData):
        super(Waveform, self).__init__()

        self.m_penColor = QColor(Qt.white)
        self.m_fillColor = QColor(Qt.black)
        self.getData = getData
        self.startTimer(66)
        self.drawBorder = False

    def childPositionChanged(self):
        self.prepareGeometryChange()

    def boundingRect(self):
        return QRectF(0, 0, 500.0, 200.0)

    def timerEvent(self, e):
        self.update()

    @pyqtProperty(QColor)
    def penColor(self):
        return QColor(self.m_penColor)

    @penColor.setter
    def penColor(self, color):
        self.m_penColor = QColor(color)

    @pyqtProperty(QColor)
    def fillColor(self):
        return QColor(self.m_fillColor)

    @fillColor.setter
    def fillColor(self, color):
        self.m_fillColor = QColor(color)

    def paint(self, painter, option, widget):

        rect = self.boundingRect()
        data = self.getData()
        w = rect.width() / data.shape[0] * 0.8
        l = rect.left() + rect.width() * 0.2 / 2
        h = rect.height() * 1.0

        # Coordinate system -- zero line
        painter.setPen(Qt.white)
        painter.drawLine(l, rect.bottom() - h/2,
                         l+data.shape[0]*w, rect.bottom() - h/2)

        path = QPainterPath()
        path.moveTo(l, rect.top() + h/2 - data[0]*h)
        for i, c in enumerate(data[1:]):
            path.lineTo(l + (i+1)*w, rect.top() + h/2 - c*h)

        painter.setPen(QPen(self.m_penColor, 2.0, Qt.SolidLine, Qt.RoundCap))
        painter.drawPath(path)

        if self.drawBorder:
            path = QPainterPath()
            path.moveTo(rect.left(), rect.top())
            path.lineTo(rect.left(), rect.bottom())
            path.lineTo(rect.right(), rect.bottom())
            path.lineTo(rect.right(), rect.top())
            path.lineTo(rect.left(), rect.top())

            painter.setPen(QPen(self.m_penColor, 5.0, Qt.SolidLine, Qt.RoundCap))
            painter.drawPath(path)


class GraphicsView(QGraphicsView):
    keyPressed = pyqtSignal(int)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.close()

        self.keyPressed.emit(Qt.Key(e.key()))


class Dialog(QDialog):
    NumGridRows = 3
    NumButtons = 4

    def __init__(self, extra, synth, decoders, set_decoder):
        super(Dialog, self).__init__()

        self.synth = synth
        self.decoders = decoders
        self.set_decoder = set_decoder

        self.createMenu()
        self.createAudioControls()
        self.createDatasetControls()
        self.createGridGroupBox()

        mainLayout = QVBoxLayout()
        mainLayout.setMenuBar(self.menuBar)
        vLayout = QVBoxLayout()
        vLayout.addWidget(self.audioControls)
        # vLayout.addWidget(self.datasetControls)
        vLayout.addWidget(self.gridGroupBox)
        hLayout = QHBoxLayout()
        vGroupBox = QGroupBox()
        vGroupBox.setLayout(vLayout)
        hLayout.addWidget(vGroupBox)
        hLayout.addWidget(extra)
        hGroupBox = QGroupBox()
        hGroupBox.setLayout(hLayout)
        hGroupBox.setFlat(True)
        title = QLabel("Sounderfeit")
        title.setStyleSheet("font-size: 20pt; font-weight: bold;")
        mainLayout.addWidget(title)
        mainLayout.addWidget(hGroupBox)
        self.setLayout(mainLayout)

        self.setWindowTitle("Sounderfeit")

    def createMenu(self):
        self.menuBar = QMenuBar()

        self.fileMenu = QMenu("&File", self)
        self.exitAction = self.fileMenu.addAction("E&xit")
        self.menuBar.addMenu(self.fileMenu)

        self.exitAction.triggered.connect(self.accept)

    def createAudioControls(self):
        self.audioControls = QGroupBox("Audio")
        vlayout = QVBoxLayout()

        layout = QHBoxLayout()
        button = QPushButton("Play")
        button.clicked.connect(lambda: self.synth.start())
        layout.addWidget(button)
        button = QPushButton("Stop")
        button.clicked.connect(lambda: self.synth.stop())
        layout.addWidget(button)

        layout2 = QVBoxLayout()
        for n,d in enumerate(self.decoders):
            button = QRadioButton(d)
            button.setChecked(n==0)
            def make_clicked(d,button):
                def clicked():
                    if button.isChecked():
                        self.set_decoder(d)
                        print('Decoder set to',d)
                        self.synth.setMode(0)
                return clicked
            button.clicked.connect(make_clicked(d,button))
            layout2.addWidget(button)
        button = QRadioButton("STK Bowed")
        button.clicked.connect(lambda: self.synth.setMode(1))
        layout2.addWidget(button)

        vlayout.addLayout(layout)
        vlayout.addLayout(layout2)
        self.audioControls.setLayout(vlayout)

    def createDatasetControls(self):
        self.datasetControls = QGroupBox("Dataset")
        layout = QHBoxLayout()
        button = QPushButton("Load")
        layout.addWidget(button)
        button = QPushButton("Train")
        layout.addWidget(button)

        self.datasetControls.setLayout(layout)

    def createGridGroupBox(self):
        self.gridGroupBox = QGroupBox("Parameters")
        layout = QGridLayout()
        volume_slider = QSlider()
        volume_slider.setRange(0, 128)
        volume_slider.setValue(64)
        position_slider = QSlider()
        position_slider.setRange(0, 64)
        position_slider.setValue(32)
        pressure_slider = QSlider()
        pressure_slider.setRange(0, 128)
        pressure_slider.setValue(64)
        latent1_slider = QSlider()
        latent1_slider.setRange(0, 128)
        latent1_slider.setValue(64)
        layout.addWidget(volume_slider,      0, 0, alignment=Qt.AlignHCenter)
        layout.addWidget(QLabel('volume'),   1, 0, alignment=Qt.AlignHCenter)
        layout.addWidget(position_slider,    0, 1, alignment=Qt.AlignHCenter)
        layout.addWidget(QLabel('position'), 1, 1, alignment=Qt.AlignHCenter)
        layout.addWidget(pressure_slider,    0, 2, alignment=Qt.AlignHCenter)
        layout.addWidget(QLabel('pressure'), 1, 2, alignment=Qt.AlignHCenter)
        layout.addWidget(latent1_slider,     0, 3, alignment=Qt.AlignHCenter)
        layout.addWidget(QLabel('latent1'),  1, 3, alignment=Qt.AlignHCenter)
        self.gridGroupBox.setLayout(layout)

        position_slider.valueChanged.connect(
            lambda: self.synth.setParam(0, position_slider.value()))

        pressure_slider.valueChanged.connect(
            lambda: self.synth.setParam(1, pressure_slider.value()))

        volume_slider.valueChanged.connect(
            lambda: self.synth.setParam(2, volume_slider.value()/128.0))

        latent1_slider.valueChanged.connect(
            lambda: self.synth.setParam(3, latent1_slider.value()/128.0))

def dialog(decoders, set_decoder, synth=None):
    import sys

    app = QApplication(sys.argv)

    cycleText = QGraphicsTextItem()
    cycleText.setHtml('<font color=\"white\"><b>One Cycle</b></font>')

    scopeText = QGraphicsTextItem()
    scopeText.setHtml('<font color=\"white\"><b>Oscilloscope</b></font>')

    spectrumText = QGraphicsTextItem()
    spectrumText.setHtml('<font color=\"white\"><b>Log Spectrum</b></font>')

    def getCycle():
        cycle = synth.lastCycle()
        cycle = detrend(lfilter([1],[-1], x=cycle))
        return cycle
    waveformCycle = Waveform(getCycle)

    lastScope = [np.zeros(10)]
    def getScope():
        scope = synth.lastScope()
        lastScope[0] = np.copy(scope)
        return scope/2
    waveformScope = Waveform(getScope)

    window = [np.zeros(10)]
    def getSpectrum():
        wave = lastScope[0]
        w = window[0]
        if w.shape[0] != wave.shape[0]:
            window[0] = hamming(wave.shape[0], False)
            w = window[0]
        spec = np.log(np.abs(np.fft.rfft(wave * w))+0.001)
        spec = (spec + 2) / 12.0
        return spec
    waveformSpectrum = Waveform(getSpectrum)

    w = cycleText.boundingRect().width()
    cycleBoundingRect = waveformCycle.mapToScene(waveformCycle.boundingRect())\
                                     .boundingRect()
    cycleText.setPos(0, cycleBoundingRect.bottom() + 10.0)

    waveformScope.setPos(0, cycleBoundingRect.bottom() + 100.0)
    w = scopeText.boundingRect().width()
    scopeBoundingRect = waveformScope.mapToScene(waveformScope.boundingRect())\
                                     .boundingRect()
    scopeText.setPos(0, scopeBoundingRect.bottom() + 10.0)

    waveformSpectrum.setPos(0, scopeBoundingRect.bottom() + 100.0)
    w = spectrumText.boundingRect().width()
    spectrumBoundingRect = waveformSpectrum.mapToScene(waveformSpectrum.boundingRect())\
                                           .boundingRect()
    spectrumText.setPos(0, spectrumBoundingRect.bottom() + 10.0)

    scene = QGraphicsScene()
    scene.addItem(waveformCycle)
    scene.addItem(waveformScope)
    scene.addItem(waveformSpectrum)
    scene.addItem(cycleText)
    scene.addItem(scopeText)
    scene.addItem(spectrumText)
    scene.setBackgroundBrush(Qt.black)

    view = GraphicsView()
    view.setRenderHints(QPainter.Antialiasing)
    view.setTransformationAnchor(QGraphicsView.NoAnchor)
    view.setScene(scene)
    view.show()
    view.setFocus()

    # Make enough room in the scene for stickman to jump and die.
    sceneRect = scene.sceneRect()
    view.resize(sceneRect.width() + 100, sceneRect.height() + 100)
    view.setSceneRect(sceneRect)

    return Dialog(view, synth, decoders, set_decoder).exec_()

if __name__ == '__main__':
    import sys
    sys.exit(dialog())
