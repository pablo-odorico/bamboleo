#!/usr/bin/env python3

import os
import sys
import json
import subprocess

try:
    from PyQt5.QtWidgets import *
    from PyQt5.QtCore import *
    from PyQt5.QtGui import *
except ImportError as e:
    print(str(e) + '\nPlease install required packages with:\n  pip3 install pyqt5')
    sys.exit(1)

import bamboleo

defaultConfig = {
    'dpi' : 400,
    'directory' : QDir.homePath(),
    'savePlots' : False
}
configPath = os.path.join(QStandardPaths.standardLocations(QStandardPaths.ConfigLocation)[0], 'bamboleo.json')

class MainDialog(QDialog):
    fileOpened = pyqtSignal(dict)

    def __init__(self, config):
        super().__init__()

        self.setWindowTitle('Bamboleo')
        self.vLayout = QVBoxLayout(self)

        self.dpi = QSpinBox(self)
        self.dpi.setRange(100, 1000)
        self.dpi.setSingleStep(50)
        self.dpi.setSuffix(' dpi')
        self.dpi.setValue(config['dpi'])
        self.hLayout = QHBoxLayout()
        self.hLayout.setContentsMargins(10, 10, 10, 0)
        self.hLayout.addWidget(QLabel('Image dots-per-inch', self))
        self.hLayout.addWidget(self.dpi)

        self.savePlots = QCheckBox('Save intermadiate plots', self)
        self.savePlots.setChecked(config['savePlots'])
        self.hLayout2 = QHBoxLayout()
        self.hLayout2.setContentsMargins(10, 0, 20, 20)
        self.hLayout2.addStretch()
        self.hLayout2.addWidget(self.savePlots)

        self.fileDialog = QFileDialog(self, Qt.Widget)
        self.fileDialog.setWindowFlags(self.fileDialog.windowFlags() & ~Qt.Dialog)
        self.fileDialog.setOption(QFileDialog.DontUseNativeDialog)
        self.fileDialog.setNameFilter('Image files (*.jpg *.jpeg *.png)')
        self.fileDialog.setDirectory(config['directory'])

        self.vLayout.setContentsMargins(0, 0, 0, 0)
        self.vLayout.addLayout(self.hLayout)
        self.vLayout.addWidget(self.fileDialog)
        self.vLayout.addLayout(self.hLayout2)

        self.fileDialog.rejected.connect(self.close)
        self.fileDialog.fileSelected.connect(self.triggerExecution)

    def triggerExecution(self, path):
        params = {
            'inputImagePath' : path,
            'outputDir' : os.path.splitext(path)[0], # Name of the input image without the extension
            'savePlots' : self.savePlots.isChecked(),
            'dpi' : self.dpi.value()
        }
        
        # Save config
        config = {
            'dpi' : self.dpi.value(),
            'directory' : self.fileDialog.directory().absolutePath(),
            'savePlots' : self.savePlots.isChecked()
        }
        open(configPath, 'w').write(json.dumps(config))

        self.close()
        self.fileOpened.emit(params)

class UiPrinter(QObject):
    '''
    This callable class injects itself inside a module to hijack the print() calls.
    When called it will append the text into a QPlainTextEdit.
    '''
    def __init__(self, module, textEdit):
        self.textEdit = textEdit
        module.print = self
    def __call__(self, *args, **kwargs):
        text = ' '.join(args)
        print('Log:', text)
        self.textEdit.appendPlainText(text)
        self.textEdit.viewport().update()

class WorkerThread(QThread):
    def __init__(self, params):
        self.params = params
        super().__init__()
    def run(self):
        bamboleo.process(**self.params)

class ExecuteDialog(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Bamboleo')    
        self.vLayout = QVBoxLayout(self)
        self.vLayout.setContentsMargins(10,10,10,10)
        self.vLayout.setSpacing(10)
        self.thread = None

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.document().setDefaultFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))

        self.vLayout.addWidget(self.log)
        self.uiPrinter = UiPrinter(bamboleo, self.log)

        self.openDirButton = QPushButton('Open Output')
        self.openDirButton.setEnabled(False)
        self.openDirButton.clicked.connect(self.openOutputDir)
        self.closeButton = QPushButton('Close')
        self.closeButton.setEnabled(False)
        self.closeButton.clicked.connect(self.close)

        self.hLayout = QHBoxLayout()
        self.hLayout.setSpacing(10)
        self.hLayout.setContentsMargins(0, 0, 0 ,0)
        self.hLayout.addWidget(self.openDirButton)
        self.hLayout.addWidget(self.closeButton)

        self.vLayout.addLayout(self.hLayout)

    def startExecution(self, params):
        self.setWindowTitle('Bamboleo (Working)')
        self.show()
        self.params = params
        self.thread = WorkerThread(params)
        self.thread.finished.connect(self.executionDone)
        self.thread.start()

    def executionDone(self):
        self.setWindowTitle('Bamboleo (Done)')
        self.openDirButton.setEnabled(True)
        self.closeButton.setEnabled(True)

    def openOutputDir(self):
        outputDir = self.params['outputDir']
        if sys.platform == "win32":
            os.startfile(outputDir)
        elif sys.platform == "darwin":
            subprocess.call(['open', outputDir])
        else:
            subprocess.call(['xdg-open', outputDir])

if __name__ == '__main__':
    try:
        config = json.loads(open(configPath, 'r').read())
        for key in defaultConfig:
            assert key in config
        print(f'Loaded config from "{configPath}"')
    except:
        print('Using default config')
        config = defaultConfig

    app = QApplication([])

    mainDialog = MainDialog(config)
    executeDialog = ExecuteDialog()
    screenRect = app.desktop().screenGeometry()
    executeDialog.resize(0.6 * screenRect.width(), 0.8 * screenRect.height()) 

    mainDialog.fileOpened.connect(executeDialog.startExecution)
    mainDialog.show()
    app.exec_()