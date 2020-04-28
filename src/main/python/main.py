from fbs_runtime.application_context.PyQt5 import ApplicationContext
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
from PyQt5 import uic
import sys
import os
from datetime import datetime
import matplotlib.pyplot as plt



class dialog_conf():
    """The class's docstring"""

    def __init__(self):
        super().__init__()

        self.Ui_DialogConf = uic.loadUi(os.path.join('src', 'build', 'ui', 'dialog_conf.ui'))
        self.Ui_DialogConf.setWindowTitle('configuration')

        # call backs for dialog here
        self.Ui_DialogConf.set_projects_button.clicked.connect(self.set_projects_path)
        self.Ui_DialogConf.set_sources_button.clicked.connect(self.set_sources_path)

    def set_projects_path(self):
        """

        :return:
        """
        import os
        import json
        from PyQt5.QtWidgets import QFileDialog
        from pytecpiv_conf import pytecpiv_get_pref

        # get the data from the conf file if exist
        file_exist, sources_path, projects_path = pytecpiv_get_pref()
        print(file_exist)

        current_directory = os.getcwd()
        print(current_directory)
        new_projects_path = QFileDialog.getExistingDirectory(self, 'Open directory', current_directory)

        if file_exist == 'yes':
            #  write in the file
            with open('pytecpiv_settings.json') as f:
                pytecpiv_settings = json.load(f)

            pytecpiv_settings['projects'] = []
            pytecpiv_settings['projects'].append({'projects_path': new_projects_path})
            with open('pytecpiv_settings.json', 'w') as outfile:
                json.dump(pytecpiv_settings, outfile)


        else:
            #  create the conf file and write in
            pytecpiv_settings = {'sources': []}
            pytecpiv_settings['sources'].append({'sources_path': ' '})
            pytecpiv_settings['projects'] = []
            pytecpiv_settings['projects'].append({'projects_path': new_projects_path})
            with open('pytecpiv_settings.json', 'w') as outfile:
                json.dump(pytecpiv_settings, outfile)

        self.sources_label.setText(sources_path)
        self.projects_label.setText(projects_path)

    def set_sources_path(self):
        """

        """
        import os
        import json
        from PyQt5.QtWidgets import QFileDialog
        from pytecpiv_conf import pytecpiv_get_pref

        # get the data from the conf file if exist
        file_exist, sources_path, projects_path = pytecpiv_get_pref()

        current_directory = os.getcwd()
        new_sources_path = QFileDialog.getExistingDirectory(self, 'Open directory', current_directory)

        if file_exist == 'yes':
            #  write in the file
            with open('pytecpiv_settings.json') as f:
                pytecpiv_settings = json.load(f)

            pytecpiv_settings['sources'] = []
            pytecpiv_settings['sources'].append({'sources_path': new_sources_path})
            with open('pytecpiv_settings.json', 'w') as outfile:
                json.dump(pytecpiv_settings, outfile)

        else:
            #  create the conf file and write in
            pytecpiv_settings = {'sources': []}
            pytecpiv_settings['sources'].append({'sources_path': new_sources_path})
            pytecpiv_settings['projects'] = []
            pytecpiv_settings['projects'].append({'projects_path': ' '})
            with open('pytecpiv_settings.json', 'w') as outfile:
                json.dump(pytecpiv_settings, outfile)

        self.sources_label.setText(sources_path)
        self.projects_label.setText(projects_path)

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

        #  define callbacks here
        self.ui_main_window.actionConfiguration.triggered.connect(self.show_conf_fn)  # menu settings
        self.dialog_conf = dialog_conf()

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

    def show_conf_fn(self):
        """This function makes visible the dialogue box for the configuration"""
        import os
        from pytecpiv_conf import pytecpiv_get_pref

        current_directory = os.getcwd()
        (file_exist, sources_path, projects_path) = pytecpiv_get_pref()
        self.dialog_conf.Ui_DialogConf.code_label.setText(current_directory)
        self.dialog_conf.Ui_DialogConf.sources_label.setText(sources_path)
        self.dialog_conf.Ui_DialogConf.projects_label.setText(projects_path)
        self.dialog_conf.Ui_DialogConf.show()

if __name__ == '__main__':
    app_context = AppContext()  # 4. Instantiate the subclass

    exit_code = app_context.run()  # 5. Invoke run()

    sys.exit(exit_code)