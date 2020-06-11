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
    """
    Class for the objects in the dialog window for configuration of pyTecPIV
    """

    def __init__(self):
        super().__init__()

        self.Ui_DialogConf = uic.loadUi(os.path.join('src', 'build', 'ui', 'dialog_conf.ui'))
        self.Ui_DialogConf.setWindowTitle('configuration')

        # call backs for dialog here
        self.Ui_DialogConf.set_projects_button.clicked.connect(self.set_projects_path)
        self.Ui_DialogConf.set_sources_button.clicked.connect(self.set_sources_path)

    def set_projects_path(self):
        """
        Defines the path where source material is generally located.
        Used to accelerate finding the project source by jumping directly into a usual directory.
        If file pytecpiv_settings.json does not exist in pyTecPIV source directory, a new file is created.
        If file exists, path is read from file if exist. Path in file is updated.
        """
        import os
        import json
        from PyQt5.QtWidgets import QFileDialog
        from pytecpiv_conf import pytecpiv_get_pref

        # get the data from the conf file if exist
        file_exist, sources_path, projects_path = pytecpiv_get_pref()

        current_directory = os.getcwd()
        new_projects_path = QFileDialog.getExistingDirectory(self.Ui_DialogConf, 'Open directory', current_directory)

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
            message = '> Creating configuration file: pytecpiv_settings.json'
            app_context.d_print(message)

            pytecpiv_settings = {'sources': []}
            pytecpiv_settings['sources'].append({'sources_path': ' '})
            pytecpiv_settings['projects'] = []
            pytecpiv_settings['projects'].append({'projects_path': new_projects_path})
            with open('pytecpiv_settings.json', 'w') as outfile:
                json.dump(pytecpiv_settings, outfile)

        self.Ui_DialogConf.sources_label.setText(sources_path)
        self.Ui_DialogConf.projects_label.setText(new_projects_path)

        message = '> New projects path written in file pytecpiv_settings.json'
        app_context.d_print(message)

        message = '> New projects path is: ' + new_projects_path
        app_context.d_print(message)

    def set_sources_path(self):
        """
        Defines the path where source material is generally located.
        Used to accelerate finding the project source by jumping directly into a usual directory.
        If file pytecpiv_settings.json does not exist in pyTecPIV source directory, a new file is created.
        If file exists, path is read from file if exist. Path in file is updated.
        """
        import os
        import json
        from PyQt5.QtWidgets import QFileDialog
        from pytecpiv_conf import pytecpiv_get_pref

        # get the data from the conf file if exist
        file_exist, sources_path, projects_path = pytecpiv_get_pref()

        current_directory = os.getcwd()
        new_sources_path = QFileDialog.getExistingDirectory(self.Ui_DialogConf, 'Open directory', current_directory)

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
            message = '> Creating configuration file: pytecpiv_settings.json'
            app_context.d_print(message)

            pytecpiv_settings = {'sources': []}
            pytecpiv_settings['sources'].append({'sources_path': new_sources_path})
            pytecpiv_settings['projects'] = []
            pytecpiv_settings['projects'].append({'projects_path': ' '})
            with open('pytecpiv_settings.json', 'w') as outfile:
                json.dump(pytecpiv_settings, outfile)

        self.Ui_DialogConf.sources_label.setText(new_sources_path)
        self.Ui_DialogConf.projects_label.setText(projects_path)

        message = '> New sources path written in file pytecpiv_settings.json'
        app_context.d_print(message)

        message = '> New sources path is: ' + new_sources_path
        app_context.d_print(message)

class AppContext(ApplicationContext):
    """

    """
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

        self.ui_main_window.new_project_menu.triggered.connect(self.new_project)  # new project
        self.ui_main_window.import_calib_dng.triggered.connect(self.import_calib_img_dng)  # import calib img dng

        #  delete log file if it exists
        t = os.path.isfile('log.txt')
        if t:
            os.remove('log.txt')

        #  startup message in log file and text browser
        message = '> '+str(datetime.now())
        self.d_print(message)

        message = '> Starting new instance of pytecpiv_app_' + version
        self.d_print(message)

        #  make first credit figure
        import numpy as np
        x = np.linspace(0, 1, num=101)
        y = np.linspace(0, 1, num=101)
        dx = x[1] - x[0]
        X, Y = np.meshgrid(x, y)
        theta = np.arctan2(Y-0.5, X-0.5)
        rho = np.sqrt((X-0.5) ** 2 + (Y-0.5) ** 2)
        u = rho * np.sin(theta)
        v = rho * np.cos(theta)

        dudx, dudy = np.gradient(u, 2) / dx
        dvdx, dvdy = np.gradient(v, 2) / dx

        omega = dvdx - dudy
        m = np.sqrt(u ** 2 + v ** 2)


        fig1 = plt.figure()
        ax1f1 = fig1.add_subplot(111)
        ax1f1.pcolor(X, Y, m)
        ax1f1.quiver(X[::10,::10],Y[::10,::10],u[::10,::10],v[::10,::10], pivot='middle')
        s1 = app_name + ' v.' + version
        s2 = 'build with Python 3 and:'
        s3 = 'numpy, scikit-image, rawpy, json, hdf5, matplotlib, pandas, pyqt'

        s5 = 'D. Boutelier, 2020'
        ax1f1.margins(0, 0, tight=True)
        ax1f1.set_ylim([-0.10, 1.1])
        ax1f1.set_xlim([-0.1, 1.1])
        ax1f1.text(0.01, 0.95, s1, fontsize=18, backgroundcolor='w', color='k', fontweight='bold')
        ax1f1.text(0.01, 0.9, s2, fontsize=10, backgroundcolor='w', color='b')
        ax1f1.text(0.01, 0.85, s3, fontsize=10, backgroundcolor='w', color='b')
        ax1f1.text(0.01, 0.8, s5, fontsize=9, backgroundcolor='w', color='b')
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

    def new_project(self):
        """

        :return:
        """
        import os
        import json
        from PyQt5.QtWidgets import QFileDialog
        from pytecpiv_conf import pytecpiv_get_pref
        from shutil import copy

        # get the data from the conf file if exist
        file_exist, sources_path, projects_path = pytecpiv_get_pref()

        # create a new directory
        this_project_path = QFileDialog.getExistingDirectory(self.ui_main_window, 'Create new project directory',
                                                             projects_path)

        # create a time stamp
        this_project_create_time = str(datetime.now())

        # get project name from directory name
        (this_project_root_path, this_project_name) = os.path.split(this_project_path)

        # create a current project_metadata file
        project_metadata = {'project': [], 'data_sources': []}
        project_metadata['project'].append({
            'project_root_path': this_project_root_path,
            'project_name': this_project_name,
            'project_create_time': this_project_create_time
        })

        with open('current_project_metadata.json', 'w') as outfile:
            json.dump(project_metadata, outfile)

        message = '> New project created: '+this_project_path
        self.d_print(message)

        # make a copy of the current metadata file in the project directory
        this_project_metadata_filename = this_project_name + '.json'
        copy('current_project_metadata.json', os.path.join(this_project_root_path, this_project_name,
                                                           this_project_metadata_filename))

    def import_calib_img_dng(self):
        """
        import dng images of calibration board
        :return:
        """
        import json
        import pandas as pd
        from PyQt5.QtWidgets import QFileDialog
        # load the json file

        with open('pytecpiv_settings.json') as f:
            pytecpiv_settings = json.load(f)

        sources = pytecpiv_settings['sources']
        sources = pd.DataFrame(sources)
        sources_path = sources['sources_path'][0]
        source_calib_path = QFileDialog.getExistingDirectory(self.ui_main_window, 'Open directory', sources_path)

        message = '> Importing dng calibration images from ' + source_calib_path
        self.d_print(message)


if __name__ == '__main__':
    app_context = AppContext()  # 4. Instantiate the subclass

    exit_code = app_context.run()  # 5. Invoke run()

    sys.exit(exit_code)