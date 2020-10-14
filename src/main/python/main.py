from fbs_runtime.application_context.PyQt5 import ApplicationContext
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
from PyQt5 import uic
import sys
import os
from datetime import datetime
import matplotlib.pyplot as plt

import time
import traceback, sys

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

# GLOBAL VARIABLES
current_dataset_name = 'credits'
dataset_index = 0

time_step = 1
time_unit = 's'
time_is_defined = False

scale = 1
phys_unit = 'mm'
scale_is_defined = False

display_settings = {}

fig1 = plt.figure()
version = ''
app_name = ''

import traceback, sys


class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        `tuple` (exctype, value, traceback.format_exc() )

    result
        `object` data returned from processing, anything

    '''
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)


class Worker(QRunnable):
    """
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    """

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()


    @pyqtSlot()
    def run(self):
        """
        Initialise the runner function with passed args, kwargs.
        """
        # Retrieve args/kwargs here; and fire processing using them
        #self.fn(*self.args, **self.kwargs)

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(
                *self.args, **self.kwargs
            )
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done


class DialogImage:
    def __init__(self):
        super().__init__()
        self.ui_dialog_image = uic.loadUi(os.path.join('src', 'build', 'ui', 'dialog_image.ui'))
        self.ui_dialog_image.setWindowTitle('image properties')

        # call back here
        self.ui_dialog_image.disp_img_checkBox.stateChanged.connect(self.change_image_prop)
        self.ui_dialog_image.img_cmap_comboBox.currentIndexChanged.connect(self.change_image_prop)
        self.ui_dialog_image.img_max_val_comboBox.currentIndexChanged.connect(self.change_image_prop)
        self.ui_dialog_image.img_min_val_comboBox.currentIndexChanged.connect(self.change_image_prop)

    def change_image_prop(self):
        import json
        from pytecpiv_display import create_fig
        global dataset_index

        current_frame_number = int(app_context.ui_main_window.frame_text.text())
        dataset_index = int(app_context.ui_main_window.Dataset_comboBox.currentIndex())
        print(dataset_index)
        if dataset_index != 0:
            this_dataset_name = app_context.ui_main_window.Dataset_comboBox.currentText()
            with open('current_project_metadata.json') as f:
                project_metadata = json.load(f)

            datasets = project_metadata['datasets']
            this_dataset = datasets[this_dataset_name]

            display_image = this_dataset['image']
            image_colormap = this_dataset['name_colormap']
            min_image_value = this_dataset['min_value_image']
            max_image_value = this_dataset['max_value_image']

            new_display_image = self.ui_dialog_image.disp_img_checkBox.isChecked()
            new_image_colormap = self.ui_dialog_image.img_cmap_comboBox.currentText()
            new_image_min_value = float(self.ui_dialog_image.img_min_val_comboBox.currentText())
            new_image_max_value = float(self.ui_dialog_image.img_max_val_comboBox.currentText())

            if new_image_max_value <= new_image_min_value:
                if new_image_max_value == 1:
                    new_image_min_value = 0.9
                    self.ui_dialog_image.img_min_val_comboBox.setCurrentText(str(new_image_min_value))
                else:
                    if new_image_min_value == 0:
                        new_image_max_value = 0.1
                        self.ui_dialog_image.img_max_val_comboBox.setCurrentText(str(new_image_max_value))
                    else:
                        new_image_max_value = new_image_min_value + 0.1
                        self.ui_dialog_image.img_max_val_comboBox.setCurrentText(str(new_image_max_value))

            if this_dataset_name in ['calibration', 'experiment']:
                if not new_display_image:
                    self.ui_dialog_image.disp_img_checkBox.setChecked(True)
                new_display_image = 'yes'
            else:
                if not new_display_image:
                    new_display_image = 'no'
                else:
                    new_display_image = 'yes'

            # enter new values in dataset
            this_dataset['image'] = new_display_image
            this_dataset['name_colormap'] = new_image_colormap
            this_dataset['min_value_image'] = new_image_min_value
            this_dataset['max_value_image'] = new_image_max_value
            datasets[this_dataset_name] = this_dataset
            project_metadata['datasets'] = datasets

            with open('current_project_metadata.json', 'w') as outfile:
                json.dump(project_metadata, outfile)

            # now redraw the image
            app_context.rm_mpl()  # clear the plotting area
            display_settings = {'dataset_name': this_dataset_name, 'frame_num': current_frame_number}
            fig1 = plt.figure()
            fig1 = create_fig(fig1, display_settings)
            app_context.add_mpl(fig1)


class DialogTime:
    def __init__(self):
        super().__init__()
        self.ui_dialog_time = uic.loadUi(os.path.join('src', 'build', 'ui', 'dialog_time.ui'))
        self.ui_dialog_time.setWindowTitle('time')

        # callback here
        self.ui_dialog_time.time_unit_comboBox.currentIndexChanged.connect(self.time_changed)
        self.ui_dialog_time.time_value_comboBox.currentIndexChanged.connect(self.time_changed)

    def time_changed(self):
        import json
        global time_unit, time_step, time_is_defined

        # read the values
        time_step = float(self.ui_dialog_time.time_value_comboBox.currentText())
        time_unit = self.ui_dialog_time.time_unit_comboBox.currentText()
        time_is_defined = True

        with open('current_project_metadata.json') as f:
            project_metadata = json.load(f)

        sources = project_metadata['data_sources']
        sources['time_interval'] = time_step
        sources['time_unit'] = time_unit
        sources['time_interval_is_defined'] = time_is_defined

        project_metadata['data_sources'] = sources

        with open('current_project_metadata.json', 'w') as outfile:
            json.dump(project_metadata, outfile)


class DialogConf:
    """
    Class for the objects in the dialog window for configuration of pyTecPIV
    """

    def __init__(self):
        super().__init__()

        self.ui_dialog_conf = uic.loadUi(os.path.join('src', 'build', 'ui', 'dialog_conf.ui'))
        self.ui_dialog_conf.setWindowTitle('configuration')

        # call backs for dialog here
        self.ui_dialog_conf.set_projects_button.clicked.connect(self.set_projects_path)
        self.ui_dialog_conf.set_sources_button.clicked.connect(self.set_sources_path)

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
        new_projects_path = QFileDialog.getExistingDirectory(self.ui_dialog_conf, 'Open directory', current_directory)
        new_projects_path = os.path.normpath(new_projects_path)

        # get value cores
        core_fraction = self.ui_dialog_conf.SliderCores.value() / 100

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

        self.ui_dialog_conf.sources_label.setText(sources_path)
        self.ui_dialog_conf.projects_label.setText(new_projects_path)

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
        new_sources_path = QFileDialog.getExistingDirectory(self.ui_dialog_conf, 'Open directory', current_directory)
        new_sources_path = os.path.normpath(new_sources_path)

        # get value cores
        core_fraction = self.ui_dialog_conf.SliderCores.value() / 100

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

        self.ui_dialog_conf.sources_label.setText(new_sources_path)
        self.ui_dialog_conf.projects_label.setText(projects_path)

        message = '> New sources path written in file pytecpiv_settings.json'
        app_context.d_print(message)

        message = '> New sources path is: ' + new_sources_path
        app_context.d_print(message)


class DialogCalibrationBoards:
    """
        Class for the objects in the dialog window for calibration boards
    """

    def __init__(self):
        super().__init__()

        self.ui_dialog_calibration_boards = uic.loadUi(os.path.join('src', 'build', 'ui',
                                                                    'dialog_calibration_boards.ui'))
        self.ui_dialog_calibration_boards.setWindowTitle('calibration boards')

        import json
        with open('calibration_boards.json') as f:
            calibration_boards = json.load(f)

        list_items = calibration_boards.keys()
        self.ui_dialog_calibration_boards.calibration_borads_comboBox.addItems(list_items)

        # call backs for dialog here
        self.ui_dialog_calibration_boards.buttonBox.accepted.connect(
            self.calibration_boards_accepted)

    def calibration_boards_accepted(self):
        """
        reads the value of selected calibration boards
        :return:
        """
        import json
        board_name = self.ui_dialog_calibration_boards.calibration_borads_comboBox.currentText()
        print(board_name)

        with open('calibration_boards.json') as f:
            calibration_boards = json.load(f)
        selected_calibration_board = calibration_boards[board_name]
        nx = selected_calibration_board['nx']
        ny = selected_calibration_board['ny']
        square_size = selected_calibration_board['square_size']
        phys_unit = selected_calibration_board['unit']

        with open('current_project_metadata.json') as f:
            project_metadata = json.load(f)

        calibration_data = project_metadata['calibration']
        calibration_data['board_name'] = board_name
        calibration_data['nx'] = nx
        calibration_data['ny'] = ny
        calibration_data['sq_size'] = square_size
        calibration_data['phys_unit'] = phys_unit

        project_metadata['calibration'] = calibration_data
        with open('current_project_metadata.json', 'w') as outfile:
            json.dump(project_metadata, outfile)


class DialogPreprocessing:
    def __init__(self):
        super().__init__()
        self.ui_dialog_preprocessing = uic.loadUi(os.path.join('src', 'build', 'ui', 'dialog_preprocessing.ui'))
        self.ui_dialog_preprocessing.setWindowTitle('preprocessing')

        # call back here
        self.ui_dialog_preprocessing.ButtonTest.clicked.connect(self.test_preprocessing)

    def test_preprocessing(self):
        bool_inverse_img = self.ui_dialog_preprocessing.checkBox_inverse.checkState()  # 0 = unchecked, 2 = checked
        bool_gaussian_blur = self.ui_dialog_preprocessing.checkBox_gaussian.checkState()

        print(bool_inverse_img)
        print(bool_gaussian_blur)

        if bool_inverse_img == 2:
            if bool_gaussian_blur == 2:
                print('not implemented yet')
            else:
                current_dataset_name = app_context.ui_main_window.Dataset_comboBox.currentText()
                current_frame_number = int(app_context.ui_main_window.frame_text.text())

                import json
                with open('current_project_metadata.json') as f:
                    project_metadata = json.load(f)
                    datasets = project_metadata['datasets']
                    this_dataset = datasets[current_dataset_name]
                    img_path = this_dataset['path_img']

                from skimage import util
                from skimage import img_as_float
                from skimage import io
                img = img_as_float(io.imread(os.path.join(img_path, 'IMG_' + str(current_frame_number).zfill(4) + '.tif')))
                inverted_img = util.invert(img)

        else:
            print('not implemented yet')


class AppContext(ApplicationContext):
    """

    """

    def __init__(self):
        super().__init__()
        self.ui_main_window = uic.loadUi(os.path.join('src', 'build', 'ui', 'gui.ui'))
        self.threadpool = QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())

    def run(self):
        global version, app_name

        #  import ui from QtDesigner ui file
        version = self.build_settings['version']
        app_name = self.build_settings['app_name']
        self.ui_main_window.setWindowTitle(app_name + ' v.' + version)

        #  set the menubar
        self.ui_main_window.menubar = self.ui_main_window.menuBar()
        self.ui_main_window.menubar.setNativeMenuBar(False)

        #  define the callbacks here
        self.ui_main_window.actionConfiguration.triggered.connect(self.show_conf_fn)  # menu settings
        self.dialog_conf = DialogConf()

        self.ui_main_window.Img_pushButton.clicked.connect(self.show_image_dialog)
        self.dialog_image = DialogImage()

        self.ui_main_window.actionDefine_time_interval.triggered.connect(self.show_time)
        self.dialog_time = DialogTime()

        self.ui_main_window.Calibration_board_menu.triggered.connect(self.show_calibration_boards)
        self.dialog_calibration_boards = DialogCalibrationBoards()

        self.ui_main_window.new_project_menu.triggered.connect(self.new_project)  # new project

        self.ui_main_window.import_calib_dng.triggered.connect(self.import_calib_img_dng)  # import calib img dng
        self.ui_main_window.import_exp_dng.triggered.connect(self.import_exp_img_dng)  # import calib img dng

        self.ui_main_window.Dataset_comboBox.currentIndexChanged.connect(self.dataset_combobox_fn)
        self.ui_main_window.FrameUpPushButton.clicked.connect(self.plus_frame)
        self.ui_main_window.FrameDownPushButton.clicked.connect(self.minus_frame)

        # selected rectification type
        self.ui_main_window.action_proj.triggered.connect(self.rectification_proj)
        self.ui_main_window.action_poly2.triggered.connect(self.rectification_poly2)
        self.ui_main_window.action_poly3.triggered.connect(self.rectification_poly3)
        self.ui_main_window.action_proj_poly3.triggered.connect(self.rectification_proj_poly3)
        self.ui_main_window.action_proj_poly2.triggered.connect(self.rectification_proj_poly2)

        # rectification
        self.ui_main_window.actionRectify.triggered.connect(self.apply_rectification)

        # dialog image intensity
        self.ui_main_window.actionImageIntensity.triggered.connect(self.show_preprocessing)
        self.dialog_image_intensity = DialogPreprocessing()

        #  delete log file if it exists
        t = os.path.isfile('log.txt')
        if t:
            os.remove('log.txt')

        #  startup message in log file and text browser
        message = '> ' + str(datetime.now())
        self.d_print(message)

        message = '> Starting new instance of pytecpiv_app_' + version
        self.d_print(message)

        #  make first credit figure
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
        s3 = 'numpy, scikit-image, rawpy, json, hdf5, matplotlib, pyqt'

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

    def apply_rectification(self):
        global current_dataset_name, dataset_index, time_step, display_settings

        current_dataset_name = self.ui_main_window.Dataset_comboBox.currentText()
        current_frame_num = int(self.ui_main_window.frame_text.text())
        import os
        import json
        with open('current_project_metadata.json') as f:
            project_metadata = json.load(f)
        datasets = project_metadata['datasets']
        this_dataset = datasets[current_dataset_name]
        img_path = this_dataset['path_img']
        starting_frame = this_dataset['starting_frame']
        number_frames = this_dataset['number_frames']

        calibration = project_metadata['calibration']
        calibration_method = calibration['method']
        calibration_function_proj = calibration['function_proj']
        calibration_function_poly = calibration['function_poly']

        self.ui_main_window.statusbar.showMessage('Correcting images. Please wait this may take a while.')

        ##### NEW
        from pytecpiv_rectif import correct_images

        #correct_images(img_path, starting_frame, number_frames, calibration_method, calibration_function_proj,
        #               calibration_function_poly)

        # start the import in a different thread in order to not freeze the GUI
        worker = Worker(correct_images(img_path, starting_frame, number_frames,
                                       calibration_method,
                                       calibration_function_proj,
                                       calibration_function_poly))

        # actions to be taken after import thread is finished.
        worker.signals.finished.connect(self.correction_thread_complete)

        # start the thread
        self.threadpool.start(worker)

        # END

    def correction_thread_complete(self):

        # remove message
        self.ui_main_window.statusbar.clearMessage()

        current_dataset_name = self.ui_main_window.Dataset_comboBox.currentText()
        current_frame_num = int(self.ui_main_window.frame_text.text())

        import os
        import json
        with open('current_project_metadata.json') as f:
            project_metadata = json.load(f)

        datasets = project_metadata['datasets']
        this_dataset = datasets[current_dataset_name]
        img_path = this_dataset['path_img']
        starting_frame = this_dataset['starting_frame']
        number_frames = this_dataset['number_frames']

        # create new dataset
        import imagesize
        img_width, img_height = imagesize.get(os.path.join(img_path, 'Corrected', 'IMG_0001.tif'))
        current_dataset = {'starting_frame': starting_frame,
                           'number_frames': number_frames,
                           'image': True,
                           'vector': False,
                           'scalar': False,
                           'path_img': os.path.join(img_path, 'Corrected'),
                           'min_value_image': 0,
                           'max_value_image': 1,
                           'name_colormap': 'gray',
                           'pixel_width': img_width,
                           'pixel_height': img_height,
                           'bit_depth': 16,
                           'image_format': 'tif'}
        this_dataset = {'experiment corrected': current_dataset}
        project_metadata['datasets'].update(this_dataset)

        with open('current_project_metadata.json', 'w') as outfile:
            json.dump(project_metadata, outfile)

        new_dataset_name = 'experiment corrected'

        # populate the display_settings dictionary
        display_settings.update({'dataset_name': new_dataset_name})

        # update and change combobox
        self.ui_main_window.Dataset_comboBox.addItem(new_dataset_name)
        new_index = self.ui_main_window.Dataset_comboBox.findText(new_dataset_name)
        self.ui_main_window.Dataset_comboBox.setCurrentIndex(new_index)

        self.ui_main_window.frame_text.setText(str(current_dataset['starting_frame']))
        self.ui_main_window.time_text.setText(str((current_dataset['starting_frame'] - 1) / time_step))

    def rectification_proj(self):
        """
        rectification with projective transformation
        :return:
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import json
        from skimage import io
        from skimage import img_as_uint, img_as_float
        from skimage import transform as tf
        from skimage.transform import warp
        import pickle
        from pytecpiv_rectif import find_control_point, find_error_proj
        import imagesize

        global dataset_index

        self.ui_main_window.Dataset_comboBox.setCurrentText('calibration')

        # create wait message
        self.ui_main_window.statusbar.showMessage("Click the four corners of the calibration board.")

        # get the calibration board
        with open('current_project_metadata.json') as f:
            project_data = json.load(f)
        calibration = project_data['calibration']
        nx = calibration['nx']
        ny = calibration['ny']
        square_size = calibration['sq_size']
        phys_unit = calibration['phys_unit']

        # user to provide the 4 corners points
        points = plt.ginput(4)
        points = np.asarray(points)
        print(points)

        # measure average distances
        F = 0.7
        LX = F * (0.5 * ((points[1, 0] - points[0, 0]) + (points[2, 0] - points[3, 0]))) / nx
        LY = F * (0.5 * ((points[0, 1] - points[3, 1]) + (points[1, 1] - points[2, 1]))) / ny
        AR = (0.5 * ((points[1, 0] - points[0, 0]) + (points[2, 0] - points[3, 0]))) / nx
        + (0.5 * ((points[0, 1] - points[3, 1]) + (points[1, 1] - points[2, 1]))) / ny
        AR = AR / square_size

        message = '> image resolution after projective transformation will be: ' + f"{AR:.2f}" + ' pixels/' + phys_unit
        self.d_print(message)

        # load image
        project = project_data['project']
        project_root_path = project['project_root_path']
        project_name = project['project_name']
        current_frame_number = int(self.ui_main_window.frame_text.text())

        img = io.imread(os.path.join(project_root_path, project_name, 'CALIB', 'IMG_' +
                                     str(current_frame_number).zfill(4)+ '.tif'))

        corrected_corners = np.zeros(points.shape)

        for i in range(0, 4):
            x = points[i, 0]
            y = points[i, 1]

            dx = LX / 2
            dy = LY / 2
            [xp, yp] = find_control_point(img, x, y, dx, dy)
            corrected_corners[i, :] = [xp, yp]

        # find the arithmetic mean of the four corners
        xm = int(0.25 * (corrected_corners[0, 0]
                         + corrected_corners[1, 0]
                         + corrected_corners[2, 0]
                         + corrected_corners[3, 0]))
        ym = int(0.25 * (corrected_corners[0, 1]
                         + corrected_corners[1, 1]
                         + corrected_corners[2, 1]
                         + corrected_corners[3, 1]))

        with open('current_project_metadata.json') as f:
            project_metadata = json.load(f)

        # define the target position of the 4 corner points relative to the mean
        W = (nx - 1) * square_size
        H = (ny - 1) * square_size
        p0 = [xm - AR * W / 2, ym + AR * H / 2]
        p1 = [xm + AR * W / 2, ym + AR * H / 2]
        p2 = [xm + AR * W / 2, ym - AR * H / 2]
        p3 = [xm - AR * W / 2, ym - AR * H / 2]
        dst = np.asarray([p0, p1, p2, p3])

        tform_proj = tf.estimate_transform('projective', dst, corrected_corners)

        f = open(os.path.join(project_root_path, project_name, 'CALIB', 'calibration_proj.pckl'), 'wb')
        pickle.dump(tform_proj, f)
        f.close()

        message = '> projective calibration function saved as: ' + os.path.join(project_root_path,
                                                                                project_name,
                                                                                'CALIB',
                                                                                'calibration_proj.pckl')
        self.d_print(message)

        # correct the calibration image
        img_warped_proj = warp(img, tform_proj)
        img_warped_proj = img_as_uint(img_warped_proj)

        #  create new RECT directory in CALIB
        calib_rect_folder = os.path.join(project_root_path, project_name, 'CALIB', 'RECT')
        if os.path.isdir(calib_rect_folder):
            message = '> Populating existing directory ' + calib_rect_folder
            self.d_print(message)
        else:
            os.makedirs(calib_rect_folder)
            message = '> Creating and populating directory ' + calib_rect_folder
            self.d_print(message)

        # create new PROJ directory in RECT
        calib_rect_proj_folder = os.path.join(project_root_path, project_name, 'CALIB', 'RECT', 'PROJ')
        if os.path.isdir(calib_rect_proj_folder):
            message = '> Populating existing directory ' + calib_rect_proj_folder
            self.d_print(message)
        else:
            os.makedirs(calib_rect_proj_folder)
            message = '> Creating and populating directory ' + calib_rect_proj_folder
            self.d_print(message)

        # save rectified image
        io.imsave(os.path.join(calib_rect_proj_folder, 'IMG_'+str(current_frame_number).zfill(4)+'.tif'),
                  img_warped_proj)

        # create dataset
        img_width, img_height = imagesize.get(os.path.join(calib_rect_proj_folder, 'IMG_' +
                                                           str(current_frame_number).zfill(4)+'.tif'))
        new_dataset = {'starting_frame': current_frame_number,
                       'number_frames': 1,
                       'image': True,
                       'vector': False,
                       'scalar': False,
                       'path_img': calib_rect_proj_folder,
                       'min_value_image': 0,
                       'max_value_image': 1,
                       'name_colormap': 'gray',
                       'pixel_width': img_width,
                       'pixel_height': img_height,
                       'bit_depth': 16,
                       'image_format': 'tif'}

        this_dataset = {'calibration corrected': new_dataset}
        project_metadata['datasets'].update(this_dataset)

        calibration['method'] = 'projective'
        calibration['function_proj'] = os.path.join(project_root_path, project_name, 'CALIB', 'calibration_proj.pckl')
        calibration['function_poly'] = ''
        calibration['corners'] = dst.tolist()
        project_metadata['calibration'] = calibration

        with open('current_project_metadata.json', 'w') as outfile:
            json.dump(project_metadata, outfile)

        # update and change combobox
        self.rm_mpl()

        new_dataset_name = 'calibration corrected'
        dataset_index = dataset_index + 1
        self.ui_main_window.Dataset_comboBox.insertItem(int(dataset_index), new_dataset_name)
        self.ui_main_window.Dataset_comboBox.setCurrentIndex(int(dataset_index))

        find_error_proj(img_warped_proj, dst, dx, dy, nx, ny)

    def rectification_poly2(self):
        """
        rectification with polynomila dgreee 2
        :return:
        """
        print('poly2')

    def rectification_poly3(self):
        """
        rectification with polynomila dgreee 3
        :return:
        """
        print('poly3')

    def rectification_proj_poly2(self):
        """
        rectification with projective transformation followed by polynomila dgreee 2
        :return:
        """
        print('proj+poly2')

    def rectification_proj_poly3(self):
        """
        rectification with projective transformation followed by polynomila dgreee 3
        :return:
        """
        print('proj+poly3')

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

    def show_preprocessing(self):
        self.dialog_image_intensity.ui_dialog_preprocessing.show()

    def show_calibration_boards(self):
        self.dialog_calibration_boards.ui_dialog_calibration_boards.show()

    def show_time(self):
        self.dialog_time.ui_dialog_time.show()

    def show_conf_fn(self):
        """This function makes visible the dialogue box for the configuration"""
        import os
        from pytecpiv_conf import pytecpiv_get_pref

        current_directory = os.getcwd()
        (file_exist, sources_path, projects_path) = pytecpiv_get_pref()
        self.dialog_conf.ui_dialog_conf.code_label.setText(current_directory)
        self.dialog_conf.ui_dialog_conf.sources_label.setText(sources_path)
        self.dialog_conf.ui_dialog_conf.projects_label.setText(projects_path)
        self.dialog_conf.ui_dialog_conf.show()

    def show_image_dialog(self):
        self.dialog_image.ui_dialog_image.show()

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

        message = '> New project created: ' + this_project_path
        self.d_print(message)

        # make a copy of the current metadata file in the project directory
        this_project_metadata_filename = this_project_name + '.json'
        copy('current_project_metadata.json', os.path.join(this_project_root_path, this_project_name,
                                                           this_project_metadata_filename))

    def import_thread_complete(self):
        import json

        global current_dataset_name, dataset_index

        self.ui_main_window.statusbar.showMessage('import completed')
        print('IMPORT THREAD FINISHED')

        with open('current_project_metadata.json') as f:
            project_metadata = json.load(f)
            datasets = project_metadata['datasets']
            current_dataset = datasets[current_dataset_name]
            num_img = current_dataset['number_frames']

        message = '> ' + str(num_img) + ' dng images imported'
        self.d_print(message)

        # update and change combobox
        self.ui_main_window.Dataset_comboBox.insertItem(int(dataset_index), current_dataset_name)
        self.ui_main_window.Dataset_comboBox.setCurrentIndex(int(dataset_index))

        self.ui_main_window.frame_text.setText(str(current_dataset['starting_frame']))
        self.ui_main_window.time_text.setText(str((current_dataset['starting_frame'] - 1) / time_step))

        self.ui_main_window.statusbar.clearMessage()

    def import_img(self, source_path, output_folder, use_cores, num_img, list_img, new_dataset_name):
        from joblib import Parallel, delayed
        from pytecpiv_import import convert_dng
        import imagesize
        import json

        Parallel(n_jobs=use_cores)(delayed(convert_dng)
                                   (frame_num, os.path.join(source_path, list_img[frame_num]),
                                    output_folder) for frame_num in range(0, num_img))

        with open('current_project_metadata.json') as f:
            project_metadata = json.load(f)

        if new_dataset_name == 'calibration':
            project_metadata['data_sources'].update({'source_calibration': source_path,
                                                     'number_calibration_images': num_img,
                                                     'calibration_image_format': 'dng'})

            img_width, img_height = imagesize.get(os.path.join(output_folder, 'IMG_0001.tif'))

            # create dataset

            current_dataset = {'starting_frame': 1,
                               'number_frames': num_img,
                               'image': True,
                               'vector': False,
                               'scalar': False,
                               'path_img': output_folder,
                               'min_value_image': 0,
                               'max_value_image': 1,
                               'name_colormap': 'gray',
                               'pixel_width': img_width,
                               'pixel_height': img_height,
                               'bit_depth': 16,
                               'image_format': 'tif'}

            this_dataset = {new_dataset_name: current_dataset}
            project_metadata['datasets'].update(this_dataset)
        elif new_dataset_name == 'experiment':
            project_metadata['data_sources'].update({'source_experiment': source_path,
                                                     'number_experiment_images': num_img,
                                                     'experiment_image_format': 'dng'})

            img_width, img_height = imagesize.get(os.path.join(output_folder, 'IMG_0001.tif'))

            # create dataset
            current_dataset = {'starting_frame': 1,
                               'number_frames': num_img,
                               'image': True,
                               'vector': False,
                               'scalar': False,
                               'path_img': output_folder,
                               'min_value_image': 0,
                               'max_value_image': 1,
                               'name_colormap': 'gray',
                               'pixel_width': img_width,
                               'pixel_height': img_height,
                               'bit_depth': 16,
                               'image_format': 'tif'}

            this_dataset = {new_dataset_name: current_dataset}
            project_metadata['datasets'].update(this_dataset)

        with open('current_project_metadata.json', 'w') as outfile:
            json.dump(project_metadata, outfile)

    def import_calib_img_dng(self):
        """
        import dng images of calibration board
        :return:
        """
        import json
        import os
        from PyQt5.QtWidgets import QFileDialog

        global current_dataset_name, dataset_index, time_step, display_settings

        # load the json file
        with open('pytecpiv_settings.json') as f:
            pytecpiv_settings = json.load(f)

        sources = pytecpiv_settings['sources']
        sources_path = sources['sources_path']
        parallel_conf = pytecpiv_settings['parallel']
        fraction_core = parallel_conf['core-fraction']

        source_calib_path = QFileDialog.getExistingDirectory(self.ui_main_window, 'Open directory', sources_path)
        source_calib_path = os.path.normpath(source_calib_path)

        # create wait message
        self.ui_main_window.statusbar.showMessage("Importing images. Please wait, this may take a while.")

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

        current_dataset_name = 'calibration'
        dataset_index = dataset_index + 1

        list_img = sorted(os.listdir(source_calib_path))  # find images in target directory
        num_img = len(list_img)  # get number of images in directory

        # get number of available core
        available_cores = os.cpu_count()
        use_cores = int(fraction_core * available_cores)

        # start the import in a different thread in order to not freeze the GUI
        worker = Worker(self.import_img, source_calib_path, calibration_folder, use_cores, num_img,
                        list_img, current_dataset_name)

        # actions to be taken after import thread is finished.
        worker.signals.finished.connect(self.import_thread_complete)

        # start the thread
        self.threadpool.start(worker)

    def import_exp_img_dng(self):
        """
        import dng images of calibration board
        :return:
        """
        import json
        import os
        from PyQt5.QtWidgets import QFileDialog

        global current_dataset_name, dataset_index, time_step, display_settings

        # load the json file
        with open('pytecpiv_settings.json') as f:
            pytecpiv_settings = json.load(f)

        sources = pytecpiv_settings['sources']
        sources_path = sources['sources_path']

        parallel_conf = pytecpiv_settings['parallel']
        fraction_core = parallel_conf['core-fraction']

        source_exp_path = QFileDialog.getExistingDirectory(self.ui_main_window, 'Open directory', sources_path)
        source_exp_path = os.path.normpath(source_exp_path)

        # create wait message
        self.ui_main_window.statusbar.showMessage("Importing images. Please wait, this may take a while.")

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

        current_dataset_name = 'experiment'
        dataset_index = dataset_index + 1

        list_img = sorted(os.listdir(source_exp_path))  # find images in target directory
        num_img = len(list_img)  # get number of images in directory

        # get number of available core
        available_cores = os.cpu_count()
        use_cores = int(fraction_core * available_cores)

        # start the import in a different thread in order to not freeze the GUI
        worker = Worker(self.import_img, source_exp_path, exp_folder, use_cores, num_img,
                        list_img, current_dataset_name)

        # actions to be taken after import thread is finished.
        worker.signals.finished.connect(self.import_thread_complete)

        # start the thread
        self.threadpool.start(worker)

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
            # print('displaying the dataset: ' + this_dataset_name)

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

            # print('displaying frame: ' + str(current_frame_number))

            self.rm_mpl()  # clear the plotting area
            display_settings = {'dataset_name': this_dataset_name, 'frame_num': current_frame_number}
            fig1 = plt.figure()
            fig1 = create_fig(fig1, display_settings)
            self.add_mpl(fig1)

    def plus_frame(self):
        import json
        from pytecpiv_display import create_fig
        global display_settings

        dataset_index = self.ui_main_window.Dataset_comboBox.currentIndex()

        if dataset_index != 0:
            this_dataset_name = self.ui_main_window.Dataset_comboBox.currentText()
            with open('current_project_metadata.json') as f:
                project_metadata = json.load(f)

            sources = project_metadata['data_sources']
            time_step = float(sources['time_interval'])
            time_unit = sources['time_unit']

            datasets = project_metadata['datasets']
            this_dataset = datasets[this_dataset_name]

            start_frame = this_dataset['starting_frame']
            end_frame = int(start_frame + this_dataset['number_frames'] - 1)

            current_frame_number = int(self.ui_main_window.frame_text.text())
            new_frame_number = int(current_frame_number + 1)

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

    def minus_frame(self):
        import json
        from pytecpiv_display import create_fig

        dataset_index = self.ui_main_window.Dataset_comboBox.currentIndex()

        if dataset_index != 0:
            this_dataset_name = self.ui_main_window.Dataset_comboBox.currentText()
            with open('current_project_metadata.json') as f:
                project_metadata = json.load(f)

            sources = project_metadata['data_sources']
            time_step = float(sources['time_interval'])
            time_unit = sources['time_unit']

            datasets = project_metadata['datasets']
            this_dataset = datasets[this_dataset_name]

            start_frame = this_dataset['starting_frame']
            end_frame = int(start_frame + this_dataset['number_frames'] - 1)

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
