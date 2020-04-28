from fbs_runtime.application_context.PyQt5 import ApplicationContext
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
from PyQt5 import uic
import sys
import os
from datetime import datetime
import matplotlib.pyplot as plt

class AppContext(ApplicationContext):
    def __init__(self):
        super().__init__()

    def run(self):
        #  import ui from QtDesigner ui file
        self.ui_main_window = uic.loadUi(os.path.join('src', 'build', 'ui', 'gui.ui'))
        version = self.build_settings['version']
        app_name = self.build_settings['app_name']
        self.ui_main_window.setWindowTitle(app_name + ' v.' + version)

        #  set the menubar
        self.ui_main_window.menubar = self.ui_main_window.menuBar()
        self.ui_main_window.menubar.setNativeMenuBar(False)

        #  delete log file if it exists
        t = os.path.isfile('log.txt')
        if t:
            os.remove('log.txt')

        #  startup message in log file and text browser
        message = str(datetime.now())
        self.d_print(message)

        message = 'Starting new instance of PyTecPIV'
        self.d_print(message)

        message = 'version ' + version
        self.d_print(message)

        #  make first credit figure
        fig1 = plt.figure()
        ax1f1 = fig1.add_subplot(111)
        s1 = app_name + ' v.' + version
        s2 = 'build on Python 3.7 with the following packages:'
        s3 = 'numpy, scikit-image, rawpy, json, hdf5, matplotlib'
        s4 = 'GUI build with Qt5'
        s5 = 'D. Boutelier, 2020'
        ax1f1.margins(0, 0, tight=True)
        ax1f1.set_ylim([0, 1])
        ax1f1.set_xlim([0, 1])
        ax1f1.text(0.01, 0.95, s1, fontsize=14)
        ax1f1.text(0.01, 0.9, s2, fontsize=10)
        ax1f1.text(0.01, 0.85, s3, fontsize=10)
        ax1f1.text(0.01, 0.775, s4, fontsize=10)
        ax1f1.text(0.01, 0.7, s5, fontsize=9)
        ax1f1.set_aspect('equal')
        ax1f1.set_axis_off()

        self.add_mpl(fig1)

        self.ui_main_window.showMaximized()
        self.ui_main_window.show()
        return self.app.exec_()

    def d_print(self, message):
        log_file = open('log.txt', 'a')
        print(message, file=log_file)
        self.ui_main_window.textBrowser.append(message)

    def add_mpl(self, fig):
        self.canvas = FigureCanvas(fig)
        self.ui_main_window.mplvl.addWidget(self.canvas)
        self.canvas.draw()

        self.figure_toolbar = NavigationToolbar(self.canvas, self.ui_main_window)
        self.ui_main_window.mplvl.addWidget(self.figure_toolbar)

    def rm_mpl(self):
        self.ui_main_window.mplvl.removeWidget(self.canvas)
        self.canvas.close()

        self.ui_main_window.mplvl.removeWidget(self.figure_toolbar)
        self.figure_toolbar.close()

if __name__ == '__main__':
    app_context = AppContext()  # 4. Instantiate the subclass

    exit_code = app_context.run()  # 5. Invoke run()

    sys.exit(exit_code)