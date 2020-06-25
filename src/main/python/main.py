from fbs_runtime.application_context.PyQt5 import ApplicationContext
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
from PyQt5 import uic
import sys
import os
from datetime import datetime
import matplotlib.pyplot as plt

# initialise global variables
current_dataset_name = 'credits'
dataset_index = 0

time_step = 1
time_unit = 's'

scale = 1
phys_unit = 'mm'

display_settings = {}

fig1 = plt.figure()
version = ''
app_name = ''


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
        new_projects_path = os.path.normpath(new_projects_path)

        # get value cores
        core_fraction = self.Ui_DialogConf.SliderCores.value() / 100

        if file_exist == 'yes':
            #  write in the file
            with open('pytecpiv_settings.json') as f:
                pytecpiv_settings = json.load(f)

            pytecpiv_settings['projects'] = {'projects_path': new_projects_path}
            pytecpiv_settings['parallel'] = {'core-fraction': core_fraction}

            with open('pytecpiv_settings.json', 'w') as outfile:
                json.dump(pytecpiv_settings, outfile)

        else:
            #  create the conf file and write in
            message = '> Creating configuration file: pytecpiv_settings.json'
            app_context.d_print(message)

            pytecpiv_settings = {'sources': {'sources_path': ' '}, 'projects': {'projects_path': new_projects_path},
                                 'parallel': {'core-fraction': core_fraction}}

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
        new_sources_path = os.path.normpath(new_sources_path)

        # get value cores
        core_fraction = self.Ui_DialogConf.SliderCores.value() / 100

        if file_exist == 'yes':
            #  write in the file
            with open('pytecpiv_settings.json') as f:
                pytecpiv_settings = json.load(f)

            pytecpiv_settings['sources'] = {'sources_path': new_sources_path}
            pytecpiv_settings['parallel'] = {'core-fraction': core_fraction}

            with open('pytecpiv_settings.json', 'w') as outfile:
                json.dump(pytecpiv_settings, outfile)

        else:
            with open('pytecpiv_settings.json') as f:
                pytecpiv_settings = json.load(f)
            #  create the conf file and write in
            message = '> Creating configuration file: pytecpiv_settings.json'
            app_context.d_print(message)

            pytecpiv_settings['sources'] = {'sources_path': new_sources_path}
            pytecpiv_settings['projects'] = {'projects_path': ' '}
            pytecpiv_settings['parallel'] = {'core-fraction': core_fraction}

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
        global version, app_name

        #  import ui from QtDesigner ui file
        self.ui_main_window = uic.loadUi(os.path.join('src', 'build', 'ui', 'gui.ui'))
        version = self.build_settings['version']
        app_name = self.build_settings['app_name']
        self.ui_main_window.setWindowTitle(app_name + ' v.' + version)

        #  set the menubar
        self.ui_main_window.menubar = self.ui_main_window.menuBar()
        self.ui_main_window.menubar.setNativeMenuBar(False)

        #  define the callbacks here
        self.ui_main_window.actionConfiguration.triggered.connect(self.show_conf_fn)  # menu settings
        self.dialog_conf = dialog_conf()

        self.ui_main_window.new_project_menu.triggered.connect(self.new_project)  # new project
        self.ui_main_window.import_calib_dng.triggered.connect(self.import_calib_img_dng)  # import calib img dng
        self.ui_main_window.import_exp_dng.triggered.connect(self.import_exp_img_dng)  # import calib img dng

        self.ui_main_window.Dataset_comboBox.currentIndexChanged.connect(self.dataset_combobox_fn)
        self.ui_main_window.FrameUpPushButton.clicked.connect(self.plus_frame)
        self.ui_main_window.FrameDownPushButton.clicked.connect(self.minus_frame)

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
        X, Y = np.meshgrid(x, y)
        theta = np.arctan2(Y-0.5, X-0.5)
        rho = np.sqrt((X-0.5) ** 2 + (Y-0.5) ** 2)
        u = rho * np.sin(theta)
        v = rho * np.cos(theta)
        m = np.sqrt(u ** 2 + v ** 2)

        fig1 = plt.figure()
        ax1f1 = fig1.add_subplot(111)
        ax1f1.pcolor(X, Y, m)
        ax1f1.quiver(X[::10, ::10], Y[::10, ::10], u[::10, ::10], v[::10, ::10], pivot='middle')
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
        this_project_path = os.path.normpath(this_project_path)

        # create a time stamp
        this_project_create_time = str(datetime.now())

        # get project name from directory name
        (this_project_root_path, this_project_name) = os.path.split(this_project_path)

        # create a current project_metadata file
        project_metadata = {'project': {'project_root_path': this_project_root_path,
                                        'project_name': this_project_name,
                                        'project_create_time': this_project_create_time},
                            'data_sources': {},
                            'datasets': {},
                            'calibration': {},
                            'preprocessing': {},
                            'PIV_settings': {}}

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
        import os
        from PyQt5.QtWidgets import QFileDialog
        from joblib import Parallel, delayed
        from pytecpiv_import import convert_dng
        import imagesize
        from tqdm import tqdm

        global current_dataset_name, dataset_index, time_step, display_settings

        # load the json file
        with open('pytecpiv_settings.json') as f:
            pytecpiv_settings = json.load(f)

        sources = pytecpiv_settings['sources']
        sources_path = sources['sources_path']

        source_calib_path = QFileDialog.getExistingDirectory(self.ui_main_window, 'Open directory', sources_path)
        source_calib_path = os.path.normpath(source_calib_path)

        parallel_conf = pytecpiv_settings['parallel']
        fraction_core = parallel_conf['core-fraction']

        message = '> Importing dng calibration images from ' + source_calib_path
        self.d_print(message)

        with open('current_project_metadata.json') as f:
            project_metadata = json.load(f)

        project_info = project_metadata['project']
        project_root_path = project_info['project_root_path']
        project_name = project_info['project_name']

        calibration_folder = os.path.join(project_root_path, project_name, 'CALIB')
        if os.path.isdir(calibration_folder):
            message = '> Populating existing directory ' + calibration_folder
            self.d_print(message)
        else:
            os.makedirs(calibration_folder)
            message = '> Creating and populating directory ' + calibration_folder
            self.d_print(message)

        list_img = sorted(os.listdir(source_calib_path))  # find images in target directory
        num_img = len(list_img)   # get number of images in directory

        # get number of available core
        available_cores = os.cpu_count()
        use_cores = int(fraction_core * available_cores)

        Parallel(n_jobs=use_cores)(delayed(convert_dng)
                                             (frame_num, os.path.join(source_calib_path, list_img[frame_num]),
                                              calibration_folder) for frame_num in tqdm(range(0, num_img)))

        message = '> ' + str(num_img) + ' dng calibration images imported'
        self.d_print(message)

        project_metadata['data_sources'].update({'source_calibration': source_calib_path,
                                                 'number_calibration_images': num_img,
                                                 'calibration_image_format': 'dng'})

        img_width, img_height = imagesize.get(os.path.join(calibration_folder, 'IMG_0001.tif'))

        # create dataset
        current_dataset = {'starting_frame': 1,
                           'number_frames': num_img,
                           'image': 'yes',
                           'vector': 'no',
                           'scalar': 'no',
                           'path_img': calibration_folder,
                           'min_value_image': 0,
                           'max_value_image': 1,
                           'name_colormap': 'gray',
                           'pixel_width': img_width,
                           'pixel_height': img_height,
                           'bit_depth': 16,
                           'image_format': 'tif'}

        this_dataset = {'calibration': current_dataset}
        project_metadata['datasets'].update(this_dataset)

        with open('current_project_metadata.json', 'w') as outfile:
            json.dump(project_metadata, outfile)

        current_dataset_name = 'calibration'
        dataset_index = dataset_index + 1

        # update and change combobox
        self.ui_main_window.Dataset_comboBox.insertItem(int(dataset_index), current_dataset_name)
        self.ui_main_window.Dataset_comboBox.setCurrentIndex(int(dataset_index))

        self.ui_main_window.frame_text.setText(str(current_dataset['starting_frame']))
        self.ui_main_window.time_text.setText(str((current_dataset['starting_frame']-1) / time_step))

    def import_exp_img_dng(self):
        """
        import dng images of calibration board
        :return:
        """
        import json
        import os
        from PyQt5.QtWidgets import QFileDialog
        from joblib import Parallel, delayed
        from pytecpiv_import import convert_dng
        import imagesize
        from tqdm import tqdm

        global current_dataset_name, dataset_index, time_step, display_settings

        # load the json file
        with open('pytecpiv_settings.json') as f:
            pytecpiv_settings = json.load(f)

        sources = pytecpiv_settings['sources']
        sources_path = sources['sources_path']

        source_exp_path = QFileDialog.getExistingDirectory(self.ui_main_window, 'Open directory', sources_path)
        source_exp_path = os.path.normpath(source_exp_path)

        parallel_conf = pytecpiv_settings['parallel']
        fraction_core = parallel_conf['core-fraction']

        message = '> Importing dng calibration images from ' + source_exp_path
        self.d_print(message)

        with open('current_project_metadata.json') as f:
            project_metadata = json.load(f)

        project_info = project_metadata['project']
        project_root_path = project_info['project_root_path']
        project_name = project_info['project_name']

        exp_folder = os.path.join(project_root_path, project_name, 'EXP')
        if os.path.isdir(exp_folder):
            message = '> Populating existing directory ' + exp_folder
            self.d_print(message)
        else:
            os.makedirs(exp_folder)
            message = '> Creating and populating directory ' + exp_folder
            self.d_print(message)

        list_img = sorted(os.listdir(source_exp_path))  # find images in target directory
        num_img = len(list_img)   # get number of images in directory

        # get number of available core
        available_cores = os.cpu_count()
        use_cores = int(fraction_core * available_cores)

        Parallel(n_jobs=use_cores)(delayed(convert_dng)
                                             (frame_num, os.path.join(source_exp_path, list_img[frame_num]),
                                              exp_folder) for frame_num in tqdm(range(0, num_img)))

        message = '> ' + str(num_img) + ' dng experimental images imported'
        self.d_print(message)

        img_width, img_height = imagesize.get(os.path.join(exp_folder,'IMG_0001.tif'))

        project_metadata['data_sources'].update({'source_exp': source_exp_path,
                                                 'number_exp_images': num_img,
                                                 'exp_image_format': 'dng',
                                                 'time_interval': time_step,
                                                 'time_unit': time_unit,
                                                 'time_interval_is_defined': 'no',
                                                 'pixel_width': img_width,
                                                 'pixel_height': img_height})
        # create dataset
        current_dataset = {'starting_frame': 1,
                           'number_frames': num_img,
                           'image': 'yes',
                           'vector': 'no',
                           'scalar': 'no',
                           'path_img': exp_folder,
                           'min_value_image': 0,
                           'max_value_image': 1,
                           'name_colormap': 'gray',
                           'pixel_width': img_width,
                           'pixel_height': img_height,
                           'bit_depth': 16,
                           'image_format': 'tif'}
        this_dataset = {'experiment': current_dataset}
        project_metadata['datasets'].update(this_dataset)

        with open('current_project_metadata.json', 'w') as outfile:
            json.dump(project_metadata, outfile)

        current_dataset_name = 'experiment'
        dataset_index = dataset_index + 1

        # populate the display_settings dictionary
        display_settings.update({'dataset_name': current_dataset_name})

        # update and change combobox
        self.ui_main_window.Dataset_comboBox.insertItem(int(dataset_index), current_dataset_name)
        self.ui_main_window.Dataset_comboBox.setCurrentIndex(int(dataset_index))

        self.ui_main_window.frame_text.setText(str(current_dataset['starting_frame']))
        self.ui_main_window.time_text.setText(str((current_dataset['starting_frame']-1) / time_step))

    def dataset_combobox_fn(self):
        import json
        global dataset_index, version, app_name
        from pytecpiv_display import create_fig

        self.rm_mpl()  # clear the plotting area

        dataset_index = self.ui_main_window.Dataset_comboBox.currentIndex()
        print(dataset_index)

        if dataset_index == 0:
            print('displaying the credits')
            #  remake first credit figure
            import numpy as np
            x = np.linspace(0, 1, num=101)
            y = np.linspace(0, 1, num=101)
            X, Y = np.meshgrid(x, y)
            theta = np.arctan2(Y - 0.5, X - 0.5)
            rho = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2)
            u = rho * np.sin(theta)
            v = rho * np.cos(theta)
            m = np.sqrt(u ** 2 + v ** 2)

            fig1 = plt.figure()
            ax1f1 = fig1.add_subplot(111)
            ax1f1.pcolor(X, Y, m)
            ax1f1.quiver(X[::10, ::10], Y[::10, ::10], u[::10, ::10], v[::10, ::10], pivot='middle')
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

        else:
            this_dataset_name = self.ui_main_window.Dataset_comboBox.currentText()
            #print('displaying the dataset: ' + this_dataset_name)

            with open('current_project_metadata.json') as f:
                project_metadata = json.load(f)

            datasets = project_metadata['datasets']
            this_dataset = datasets[this_dataset_name]

            start_frame = this_dataset['starting_frame']
            end_frame = this_dataset['starting_frame']

            current_frame_number = int(self.ui_main_window.frame_text.text())

            if current_frame_number < start_frame:
                current_frame_number = start_frame

            if current_frame_number > end_frame:
                current_frame_number = end_frame

            #print('displaying frame: ' + str(current_frame_number))

            self.rm_mpl()  # clear the plotting area
            display_settings = {'dataset_name': this_dataset_name, 'frame_num': current_frame_number}
            fig1 = plt.figure()
            fig1 = create_fig(fig1, display_settings)
            self.add_mpl(fig1)

    def plus_frame(self):
        import json
        from pytecpiv_display import create_fig
        global time_step

        dataset_index = self.ui_main_window.Dataset_comboBox.currentIndex()

        if dataset_index != 0:
            this_dataset_name = self.ui_main_window.Dataset_comboBox.currentText()
            with open('current_project_metadata.json') as f:
                project_metadata = json.load(f)

            datasets = project_metadata['datasets']
            this_dataset = datasets[this_dataset_name]

            start_frame = this_dataset['starting_frame']
            end_frame = int(start_frame + this_dataset['number_frames'] -1)

            current_frame_number = int(self.ui_main_window.frame_text.text())
            new_frame_number = int(current_frame_number + 1)

            if new_frame_number < start_frame:
                new_frame_number = start_frame

            if new_frame_number > end_frame:
                new_frame_number = end_frame

            if new_frame_number != current_frame_number:
                self.ui_main_window.frame_text.setText(str(new_frame_number))
                self.ui_main_window.time_text.setText(str((new_frame_number-1) * time_step))
                self.rm_mpl()  # clear the plotting area
                display_settings = {'dataset_name': this_dataset_name, 'frame_num': new_frame_number}
                fig1 = plt.figure()
                fig1 = create_fig(fig1, display_settings)
                self.add_mpl(fig1)

    def minus_frame(self):
        import json
        from pytecpiv_display import create_fig

        dataset_index = self.ui_main_window.Dataset_comboBox.currentIndex()

        if dataset_index != 0:
            this_dataset_name = self.ui_main_window.Dataset_comboBox.currentText()
            with open('current_project_metadata.json') as f:
                project_metadata = json.load(f)

            datasets = project_metadata['datasets']
            this_dataset = datasets[this_dataset_name]

            start_frame = this_dataset['starting_frame']
            end_frame = int(start_frame + this_dataset['number_frames'] -1)

            current_frame_number = int(self.ui_main_window.frame_text.text())
            new_frame_number = int(current_frame_number - 1)

            if new_frame_number < start_frame:
                new_frame_number = start_frame

            if new_frame_number > end_frame:
                new_frame_number = end_frame

            if new_frame_number != current_frame_number:
                self.ui_main_window.frame_text.setText(str(new_frame_number))
                self.ui_main_window.time_text.setText(str((new_frame_number - 1) * time_step))
                self.rm_mpl()  # clear the plotting area
                display_settings = {'dataset_name': this_dataset_name, 'frame_num': new_frame_number}
                fig1 = plt.figure()
                fig1 = create_fig(fig1, display_settings)
                self.add_mpl(fig1)


if __name__ == '__main__':
    app_context = AppContext()  # 4. Instantiate the subclass
    exit_code = app_context.run()  # 5. Invoke run()
    sys.exit(exit_code)