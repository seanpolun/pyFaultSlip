# MIT License
#
# Copyright (c) 2020 Sean G Polun (University of Missouri)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import Qt
import sys
from MainWindow_2d import Ui_MainWindow
import faultslipMain
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import data_model
mpl.use('Qt5Agg')


class IOModel(QtCore.QAbstractListModel):
    """
    Class for managing data for 2D GUI

    Attributes:
        inputs : dict
        outputs : dict

    """
    def __init__(self, *args, inputs=None, outputs=None, **kwargs):
        super(IOModel, self).__init__(*args, **kwargs)
        self.inputs = inputs or dict()
        self.outputs = outputs or dict()
        self.input_model = data_model.ModelInputs(self.inputs)

        self.default_stress_state = "Strike-Slip"
        self.inputs['stress'] = self.default_stress_state
        self.default_shmax = self.input_model.ShMaxSS
        self.default_shmin = self.input_model.ShMinSS
        self.failure_cutoff = 0.5
        self.run_type = 'mc'
        self.inputs['mode'] = 'mc'

        # initialize defaults
        # self.inputs['flag'] = 'mc'
        # self.inputs['dip'] = 90
        # self.inputs['dipunc'] = 10
        # self.inputs['mu'] = 0.6
        # self.inputs['mu_unc'] = 0.1
        # self.inputs['pf'] = 75.0
        # self.inputs['fail_percent'] = 50
        # self.inputs['shmaxaz'] = 0.


    def json_load(self, infile):
        """
        Loads JSON file with saved parameters

        Parameters
        ----------
        infile : file (str)
            input JSON file

        Returns
        -------

        """
        with open(infile) as load_file:
            j_data = json.load(load_file)
        self.inputs = j_data['input_data'][0]
        self.input_model = data_model.ModelInputs(self.inputs)

    def json_save(self, outfile):
        """
        Saves JSON file with input parameters

        Parameters
        ----------
        outfile : str
            Path to JSON file for saving

        Returns
        -------

        """
        outdict = dict()
        outdict['input_data'] = [self.inputs]
        with open(outfile, 'w') as dump_file:
            json.dump(outdict, dump_file, indent=4, sort_keys=True)

    def update_input_model(self):
        """

        Returns
        -------

        """
        self.input_model = data_model.ModelInputs(self.inputs)


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.axcb = plt.colorbar(mpl.cm.ScalarMappable(cmap=plt.get_cmap('jet_r'), norm=mpl.colors.Normalize(vmin=0,
                                                                                                             vmax=50)))
        super(MplCanvas, self).__init__(self.fig)


def switch_unc_type(widget_obj, data_object, reference_widget, unc_type):
    if unc_type == "%":
        widget_obj.setValue(data_object.std_perc_100())
        widget_obj.setRange(0., 100.)
        widget_obj.setSingleStep(1.)
        widget_obj.update()
        # print(widget_obj.value())
        print(data_object.std_perc)
        print(data_object.std_unit())
    elif unc_type.strip() == data_object.unit_name:
        widget_obj.setValue(data_object.std_unit())
        widget_obj.setRange(reference_widget.minimum(), reference_widget.maximum())
        widget_obj.setSingleStep(reference_widget.singleStep())
        widget_obj.update()
        print(data_object.std_perc)
        print(data_object.std_unit())


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.model = IOModel()
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.orient_canvas = MplCanvas(self, width=1, height=1, dpi=72)
        # self.orient_canvas.axes = self.orient_canvas.fig.add_axes([0, 0, 1, 1])
        # self.orient_canvas.axes.set(frame_on=False)
        # self.orient_canvas.axes.set_xticks([])
        # self.orient_canvas.axes.set_yticks([])
        self.stress_orient_layout.addWidget(self.orient_canvas)
        self.init_stress()
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.plotLayout.addWidget(self.toolbar)
        self.plotLayout.addWidget(self.canvas)
        self.active_shmax = self.model.default_shmax
        self.active_shmin = self.model.default_shmin
        self.active_stress = self.model.default_stress_state

        # setup input boxes
        self.depth_mean_box.valueChanged.connect(self.depth_changed)

        self.sv_mean_box.valueChanged.connect(self.sv_mean_changed)
        self.sv_std_box.valueChanged.connect(self.sv_std_changed)
        self.sv_unc_type.currentIndexChanged.connect(self.sv_unc_type_changed)

        self.stress_type_box.currentIndexChanged.connect(self.stress_type_box_changed)

        self.shmax_mean_box.valueChanged.connect(self.shmax_mean_changed)
        self.shmax_std_box.valueChanged.connect(self.shmax_std_changed)
        self.shmax_unc_type.currentIndexChanged.connect(self.shmax_unc_type_changed)

        self.shmin_mean_box.valueChanged.connect(self.shmin_mean_changed)
        self.shmin_std_box.valueChanged.connect(self.shmin_std_changed)
        self.shmin_unc_type.currentIndexChanged.connect(self.shmin_unc_type_changed)

        self.shmax_az_mean_box.valueChanged.connect(self.shmax_az_mean_changed)
        self.shmax_az_std_box.valueChanged.connect(self.shmax_az_std_changed)
        self.az_unc_type.currentIndexChanged.connect(self.az_unc_type_changed)

        self.shmin_az_mean_box.valueChanged.connect(self.shmin_az_mean_changed)
        self.shmin_az_std_box.valueChanged.connect(self.shmin_az_std_changed)
        self.az_unc_type2.currentIndexChanged.connect(self.az_unc_type2_changed)

        self.sv_inc_unc_box.valueChanged.connect(self.sv_inc_unc_changed)
        # self.inc_unc_type.currentIndexChanged.connect(self.inc_unc_type_changed)

        self.dip_mean_box.valueChanged.connect(self.dip_mean_changed)
        self.dip_std_box.valueChanged.connect(self.dip_std_changed)
        self.dip_unc_type.currentIndexChanged.connect(self.dip_unc_type_changed)

        self.mu_mean_box.valueChanged.connect(self.mu_mean_changed)
        self.mu_std_box.valueChanged.connect(self.mu_std_changed)

        self.max_pf_box.valueChanged.connect(self.max_pf_changed)

        self.hydro_gradient_box.valueChanged.connect(self.hydro_gradient_changed)

        self.fail_percentile_box.valueChanged.connect(self.fail_percentile_changed)

        # self.det_model_yes.toggled.connect(self.det_model_enabled)
        # self.prob_model_yes.toggled.connect(self.prob_model_enabled)
        self.det_model_yes.toggled.connect(self.model_state_enabled)
        self.prob_model_yes.toggled.connect(self.model_state_enabled)

        # file dialog box
        self.file_browse_button.clicked.connect(self.shapefile_browse)
        self.shapefile_edit.textEdited.connect(self.shapefile_edit_text)

        # execute button
        self.executeButton.clicked.connect(self.execute_model)

        # menubar
        self.actionOpen_json_file.triggered.connect(self.open_json_settings)
        self.actionSave_json_settings.triggered.connect(self.save_json_settings)

        # Default Uncertainty mode (defaults to %) can be set to "percent" or "absolute"
        self._default_uncertaintyMode = "%"
        self.sv_unc_type_value = self._default_uncertaintyMode
        self.shmax_unc_type_value = self._default_uncertaintyMode
        self.shmin_unc_type_value = self._default_uncertaintyMode
        self.az_unc_type_value = self._default_uncertaintyMode
        self.az_unc_type2_value = self._default_uncertaintyMode
        # self.inc_unc_type_value = self._default_uncertaintyMode
        self.dip_unc_type_value = self._default_uncertaintyMode

        # self.update_boxes()

    def init_stress(self):
        rot_angle = self.model.input_model.ShMaxAz.mean
        fig = self.orient_canvas.fig
        # ax = fig.axes
        # fig.ax = self.orient_canvas.fig.axes
        # fig.ax = fig.add_axes([0, 0, 1, 1])
        # self.orient_canvas.fig.ax = fig.ax
        str_img = plt.imread('./resources/h_stresses.png')
        stress_im = fig.axes[0].imshow(str_img)
        midx = str_img.shape[0] / 2
        midy = str_img.shape[1] / 2
        transf = mpl.transforms.Affine2D().rotate_deg_around(midx, midy, rot_angle) + fig.axes[0].transData
        stress_im.set_transform(transf)
        fig.axes[0].set_xticks([])
        fig.axes[0].set_yticks([])
        fig.axes[0].set(frame_on=False)
        self.show()

    def update_stress(self):
        rot_angle = self.model.input_model.ShMaxAz.mean
        fig = self.orient_canvas.fig
        self.orient_canvas.fig.axes[0].cla()
        # ax = fig.axes
        # ax = fig.ax
        str_img = plt.imread('./resources/h_stresses.png')
        stress_im = fig.axes[0].imshow(str_img)
        midx = str_img.shape[0] / 2
        midy = str_img.shape[1] / 2
        transf = mpl.transforms.Affine2D().rotate_deg_around(midx, midy, rot_angle) + fig.axes[0].transData
        stress_im.set_transform(transf)
        fig.axes[0].set_xticks([])
        fig.axes[0].set_yticks([])
        fig.axes[0].set(frame_on=False)
        self.orient_canvas.fig.canvas.draw_idle()

    def depth_changed(self):
        self.model.inputs['depth'] = self.depth_mean_box.value()
        self.model.update_input_model()
        # print(self.model.inputs['depth'])

    def sv_mean_changed(self):
        self.model.input_model.Sv.mean = self.sv_mean_box.value()
        self.model.update_input_model()

    def sv_std_changed(self):
        self.model.inputs['sv_unc'] = self.sv_std_box.value()
        # self.model.update_input_model()
        if self.sv_unc_type_value == "%":
            self.model.input_model.Sv.perc_100_to_perc(self.sv_std_box.value())
        elif self.sv_unc_type_value == self.model.input_model.Sv.unit_name:
            self.model.input_model.Sv.unit_to_perc(self.sv_std_box.value())

    def sv_unc_type_changed(self):
        switch_unc_type(self.sv_std_box, self.model.input_model.Sv, self.sv_mean_box, self.sv_unc_type.currentText())
        # self.update_boxes()
        self.sv_unc_type_value = self.sv_unc_type.currentText()
        # if self.sv_unc_type.currentText() == "%":
        #     self.sv_std_box.setValue(self.model.input_model.Sv.std_perc_100)
        #     self.sv_std_box.setRange(0., 100.)
        #     self.sv_std_box.setSingleStep(1.)
        # else:
        #     self.sv_std_box.setValue(self.model.input_model.Sv.std_unit())
        #     self.sv_std_box.setRange(self.sv_mean_box.minimum(), self.sv_mean_box.maximum())
        #     self.sv_std_box.setSingleStep(self.sv_mean_box.singleStep())

    def stress_type_box_changed(self):
        # TODO: Fix odd behavior with the stress type 2D GUI box currently crashing with normal stress
        if self.stress_type_box.currentText() == "Reverse":
            self.active_shmax = self.model.input_model.ShMaxR
            self.active_shmin = self.model.input_model.ShMinR
            self.active_stress = self.stress_type_box.currentText()
            print(self.active_shmax.mean)

        elif self.stress_type_box.currentText() == "Strike-Slip":
            self.active_shmax = self.model.input_model.ShMaxSS
            self.active_shmin = self.model.input_model.ShMinSS
            self.active_stress = self.stress_type_box.currentText()
            print(self.active_shmax.mean)

        elif self.stress_type_box.currentText() == "Normal":
            self.active_shmax = self.model.input_model.ShMaxSS
            self.active_shmin = self.model.input_model.ShMinSS
            self.active_stress = self.stress_type_box.currentText()
            print(self.active_shmax.mean)

        else:
            warn = QMessageBox()
            warn.setIcon(QMessageBox.Warning)
            warn.setText("Stress State not resolved!")
            warn.setTitle("Warning")
            warn.exec_()
        self.model.inputs['stress'] = self.active_stress

    def shmax_mean_changed(self):
        self.active_shmax.mean = self.shmax_mean_box.value()
        self.model.inputs['shmax'] = self.shmax_mean_box.value()
        # self.model.update_input_model()

    def shmax_std_changed(self):
        self.model.inputs['shMunc'] = self.shmax_std_box.value()
        # self.model.update_input_model()
        if self.shmax_unc_type_value == "%":
            self.active_shmax.perc_100_to_perc(self.shmax_std_box.value())
        else:
            self.active_shmax.unit_to_perc(self.shmax_std_box.value())

    def shmax_unc_type_changed(self):
        # if self.shmax_unc_type.currentText() == "%":
        #     self.shmax_std_box.setValue(self.active_shmax.std_perc_100())
        #     self.shmax_std_box.setRange(0., 100.)
        #     self.shmax_std_box.setSingleStep(1.)
        # else:
        #     self.shmax_std_box.setValue(self.active_shmax.std_unit())
        #     self.shmax_std_box.setRange(self.shmax_mean_box.minimum(), self.shmax_mean_box.maximum())
        #     self.shmax_std_box.setSingleStep(self.shmax_mean_box.singleStep())
        switch_unc_type(self.shmax_std_box, self.active_shmax, self.shmax_mean_box, self.shmax_unc_type.currentText())
        self.shmax_unc_type_value = self.shmax_unc_type.currentText()

    def shmin_mean_changed(self):
        self.active_shmin.mean = self.shmin_mean_box.value()
        self.model.inputs['shmin'] = self.shmin_mean_box.value()
        # self.model.update_input_model()

    def shmin_std_changed(self):
        self.model.inputs['shmiunc'] = self.shmin_std_box.value()
        # self.model.update_input_model()
        if self.shmin_unc_type_value == "%":
            self.active_shmin.std_perc = self.shmin_std_box.value()
        else:
            self.active_shmin.std_perc = self.shmin_std_box.value() / self.shmin_mean_box.value()

    def shmin_unc_type_changed(self):
        # if self.shmin_unc_type.currentText() == "%":
        #     self.shmin_std_box.setValue(self.active_shmin.std_perc_100())
        # else:
        #     self.shmin_std_box.setValue(self.active_shmin.std_unit())
        switch_unc_type(self.shmin_std_box, self.active_shmin, self.shmin_mean_box, self.shmin_unc_type.currentText())
        self.shmin_unc_type_value = self.shmin_unc_type.currentText()

    def shmax_az_mean_changed(self):
        self.model.input_model.ShMaxAz.mean = self.shmax_az_mean_box.value()
        self.model.inputs['shmaxaz'] = self.shmax_az_mean_box.value()
        # self.update_stress()

    def shmin_az_mean_changed(self):
        self.model.input_model.ShMinAz.mean = self.shmin_az_mean_box.value()
        self.model.inputs['shminaz'] = self.shmin_az_mean_box.value()

    def shmax_az_std_changed(self):
        if self.az_unc_type_value == "%":
            self.model.input_model.ShMaxAz.std_perc = self.shmax_az_std_box.value()
        else:
            self.model.input_model.ShMaxAz.std_perc = self.shmax_az_std_box.value() / self.shmax_az_mean_box.value()
        self.model.inputs['azunc'] = self.shmax_az_std_box.value()
        if self.shmin_az_std_box.value() != self.shmax_az_std_box.value():
            self.shmin_az_std_box.setValue(self.shmax_az_std_box.value())

    def az_unc_type_changed(self):
        # if self.az_unc_type.currentText() == "%":
        #     self.shmax_std_box.setValue(self.model.input_model.ShMaxAz.std_perc)
        # else:
        #     self.shmax_std_box.setValue(self.model.input_model.ShMaxAz.std_unit())
        switch_unc_type(self.shmax_az_std_box, self.model.input_model.ShMaxAz, self.shmax_az_mean_box,
                        self.az_unc_type_value.currentText())
        self.az_unc_type_value = self.az_unc_type.currentText()
        if self.az_unc_type.currentText() != self.az_unc_type2.currentText():
            self.az_unc_type2.setCurrentText(self.az_unc_type.currentText())

    def shmin_az_std_changed(self):
        self.model.inputs['azunc'] = self.shmin_az_std_box.value()
        if self.az_unc_type_value == "%":
            self.model.input_model.ShMinAz.std_perc = self.shmin_az_std_box.value()
        else:
            self.model.input_model.ShMinAz.std_perc = self.shmin_az_std_box.value() / self.shmin_az_mean_box.value()
        if self.shmin_az_std_box.value() != self.shmax_az_std_box.value():
            self.shmax_az_std_box.setValue(self.shmin_az_std_box.value())

    def az_unc_type2_changed(self):
        # if self.az_unc_type2.currentText() == "%":
        #     self.shmin_std_box.set_value(self.model.input_model.ShMinAz.std_perc)
        # else:
        #     self.shmin_std_box.setValue(self.model.input_model.ShMinAz.std_unit())
        switch_unc_type(self.shmin_az_std_box, self.model.input_model.ShMinAz, self.shmin_az_mean_box,
                        self.az_unc_type2_value.currentText())
        self.az_unc_type2_value = self.az_unc_type.currentText()
        if self.az_unc_type2.currentText() != self.az_unc_type.currentText():
            self.az_unc_type.setCurrentText(self.az_unc_type2.currentText())

    def sv_inc_unc_changed(self):
        self.model.inputs['inc_unc'] = self.sv_inc_unc_box.value()

    def dip_mean_changed(self):
        self.model.input_model.Dip.mean = self.dip_mean_box.value()
        self.model.inputs['dip'] = self.dip_mean_box.value()

    def dip_std_changed(self):
        if self.dip_unc_type_value == "%":
            self.model.input_model.Dip.std_perc = self.dip_std_box.value()
        else:
            self.model.input_model.Dip.std_perc = self.dip_std_box.value() / self.dip_mean_box.value()
        self.model.inputs['dipunc'] = self.dip_std_box.value()

    def dip_unc_type_changed(self):
        # if self.dip_unc_type.currentText() == "%":
        #     self.dip_std_box.setValue(self.model.input_model.Dip.std_perc)
        # else:
        #     self.dip_std_box.setValue(self.model.input_model.Dip.std_unit())
        switch_unc_type(self.dip_std_box, self.model.input_model.Dip, self.dip_mean_box,
                        self.dip_unc_type.currentText())
        self.dip_unc_type_value = self.dip_unc_type.currentText()

    def mu_mean_changed(self):
        self.model.inputs['mu'] = self.mu_mean_box.value()
        self.model.input_model.Mu.mean = self.mu_mean_box.value()

    def mu_std_changed(self):
        self.model.inputs['mu_unc'] = self.mu_std_box.value()
        self.model.input_model.Mu.std_perc = self.mu_std_box.value()

    def max_pf_changed(self):
        self.model.inputs['pf'] = self.max_pf_box.value()
        self.model.input_model.max_pf = self.max_pf_box.value()

    def hydro_gradient_changed(self):
        self.model.input_model.SHydro.mean = self.hydro_gradient_box.value()
        self.model.inputs['hydro'] = self.hydro_gradient_box.value()

    def fail_percentile_changed(self):
        int_fail_perc = self.fail_percentile_box.value()
        self.model.inputs['fail_percent'] = int_fail_perc / 100.
        self.model.failure_cutoff = int_fail_perc / 100.
        if isinstance(self.model.outputs, data_model.Results2D):
            self.make_plot()

    # def det_model_enabled(self):
    #     if self.det_model_yes.isChecked():
    #         self.model.inputs['mode'] = 'det'
    #         self.model.run_type = 'det'
    #     # if self.prob_model_yes.isChecked():
    #     #     self.prob_model_yes.setChecked(False)
    #     print(self.model.inputs['mode'])
    #     # print(self.model.run_type)
    #
    # def prob_model_enabled(self):
    #     if self.prob_model_yes.isChecked():
    #         self.model.inputs['mode'] = 'mc'
    #         self.model.run_type = 'mc'
    #     # if self.det_model_yes.isChecked():
    #     #     self.det_model_yes.setChecked(False)
    #     print(self.model.inputs['mode'])
    #     # print(self.model.run_type)

    def model_state_enabled(self):
        button = self.sender()
        if button.text() == "Deterministic":
            if button.isChecked():
                self.model.inputs['mode'] = 'det'
                self.model.run_type = 'det'
                print(self.model.run_type)
        if button.text() == "Probabilistic":
            if button.isChecked():
                self.model.inputs['mode'] = 'mc'
                self.model.run_type = 'mc'
                print(self.model.run_type)

    def shapefile_browse(self):
        dlg = QtWidgets.QFileDialog()
        # dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        shp_filter = "Shapefile (*.shp)"
        dlg.setNameFilter(shp_filter)
        filenames = str()

        if dlg.exec_():
            filenames = dlg.selectedFiles()
            # print(filenames)
            filenames = filenames[0]
            self.model.inputs['shapefile'] = filenames
            self.shapefile_edit.setText(filenames)

    def shapefile_edit_text(self):
        self.model.inputs['shapefile'] = self.shapefile_edit.text()

    def make_plot(self):
        # initialize model data
        outputs = self.model.outputs
        input_parameters = self.model.inputs
        model_input = self.model.input_model
        out_features = self.model.outputs
        self.canvas.axes.clear()
        self.canvas.axcb.remove()

        xmin = out_features.xmin
        xmax = out_features.xmax
        ymin = out_features.ymin
        ymax = out_features.ymax
        plotmin = out_features.plotmin
        plotmax = out_features.plotmax
        cutoff = out_features.cutoff
        flag = input_parameters['mode']
        # xmin = plot_bounds[0]
        # xmax = plot_bounds[2]
        # ymin = plot_bounds[1]
        # ymax = plot_bounds[3]
        # increase bounds by set percentage
        increase_scale = 0.25
        xrange = xmax - xmin
        yrange = ymax - ymin
        xinc = (xrange * increase_scale) / 2
        yinc = (yrange * increase_scale) / 2
        xmin1 = xmin - xinc
        xmax1 = xmax + xinc
        ymin1 = ymin - yinc
        ymax1 = ymax + yinc

        if flag == 'det':
            plotmin1 = plotmin - 0.01
            plotmax1 = plotmax + 0.01
            norm1 = mpl.colors.Normalize(vmin=plotmin1, vmax=plotmax1)
        elif flag == 'mc':
            plotmin1 = plotmin - 1
            plotmax1 = plotmax + 1
            # plotmin1 = 0.
            # plotmax1 = 5.
            norm1 = mpl.colors.Normalize(vmin=plotmin1, vmax=plotmax1)
        else:
            raise ValueError("Processing mode is not defined.")
        fig = self.canvas.fig
        ax = self.canvas.axes
        ax.set_xlim(xmin1, xmax1)
        ax.set_ylim(ymin1, ymax1)
        for line in out_features.lines:
            # plot_data = np.array(out_features[i])
            # segs = np.unique([obj.seg_id for obj in line])
            n_seg = len(np.unique([obj.seg_id for obj in line]))
            segments = np.empty((n_seg, 2, 2))
            for j in range(n_seg):
                seg = line[j]
                p1 = np.asarray(seg.p1)
                p2 = np.asarray(seg.p2)
                points = np.vstack([p1, p2])
                # points = np.reshape(plot_data[j, 1:5], (2, 2))
                segments[j] = points
            # x1 = plot_data[:, 1].T
            # y1 = plot_data[:, 2].T
            # points = np.array([x1, y1]).T.reshape(-1, 1, 2)
            # segments = np.concatenate([points[:-1], points[1:]], axis=1)
            if flag == 'det':
                slip_tendency = [obj.result for obj in line]
                # slip_tendency = plot_data[:, 5]
            elif flag == 'mc':
                # slip_tendency = plot_data[:, 5]
                slip_tendency = [obj.ecdf_cutoff(cutoff) for obj in line]
            else:
                raise ValueError("Processing mode is not defined.")

            lc = mpl.collections.LineCollection(segments, cmap=plt.get_cmap('jet_r'), norm=norm1)
            lc.set_array(np.array(slip_tendency))
            lc.set_linewidth(2)
            ax.add_collection(lc)
        self.canvas.axcb = fig.colorbar(lc)
        if flag == 'det':
            self.canvas.axcb.set_label('Slip Tendency')
        elif flag == 'mc':
            self.canvas.axcb.set_label('Failure Pressure [MPa]')
        self.canvas.draw()
        self.show()

    def execute_model(self):
        input_model = self.model.input_model
        model_params = self.model.inputs
        infile = self.model.inputs['shapefile']
        # type_flag = inputs['flag']
        # model_params = {"depth": inputs['depth'], 'mode': type_flag, 'stress': self.active_stress,
        #                 'fail_percent': self.model.inputs['fail_percent']}
        results = faultslipMain.slip_tendency_2d(infile, input_model, model_params, dump_for_fsp=False)
        self.model.outputs = results
        self.make_plot()

    def update_boxes(self):
        self.depth_mean_box.setValue(self.model.inputs['depth'])

        self.sv_mean_box.setValue(self.model.input_model.Sv.mean)
        if self.sv_unc_type_value == "%":
            self.sv_std_box.setValue(self.model.input_model.Sv.std_perc_100())
        else:
            self.sv_std_box.setValue(self.model.input_model.Sv.std_unit())

        self.shmax_mean_box.setValue(self.active_shmax.mean)
        if self.shmax_unc_type_value == "%":
            self.shmax_std_box.setValue(self.active_shmax.std_perc_100())
        else:
            self.shmax_std_box.setValue(self.active_shmax.std_unit())

        self.shmin_mean_box.setValue(self.active_shmin.mean)
        if self.az_unc_type2_value == "%":
            self.shmin_std_box.setValue(self.active_shmin.std_perc_100())
        else:
            self.shmin_std_box.setValue(self.active_shmin.std_unit())

        self.shmax_az_mean_box.setValue(self.model.input_model.ShMaxAz.mean)
        self.shmin_az_mean_box.setValue(self.model.input_model.ShMinAz.mean)
        if self.az_unc_type_value == "%":
            self.shmax_az_std_box.setValue(self.model.input_model.ShMaxAz.std_perc_100())
        else:
            self.shmax_az_std_box.setValue(self.model.input_model.ShMaxAz.std_unit())

        # if self.inc_unc_type_value == "%":
        #    self.sv_inc_unc_box.setValue(self.model.inputs['inc_unc'])

        self.dip_mean_box.setValue(self.model.input_model.Dip.mean)
        if self.dip_unc_type_value == "%":
            self.dip_std_box.setValue(self.model.input_model.Dip.std_perc_100())
        else:
            self.dip_std_box.setValue(self.model.input_model.Dip.std_unit())
        self.mu_mean_box.setValue(self.model.input_model.Mu.mean)
        self.mu_std_box.setValue(self.model.input_model.Mu.std_perc)
        self.max_pf_box.setValue(self.model.input_model.max_pf)
        self.hydro_gradient_box.setValue(self.model.input_model.SHydro.mean)
        self.fail_percentile_box.setValue(self.model.inputs['fail_percent'])
        self.update_stress()

    def open_json_settings(self):
        dlg = QtWidgets.QFileDialog()
        json_filter = "JSON File (*.json)"
        dlg.setNameFilter(json_filter)
        json_file = str()
        if dlg.exec_():
            json_file = dlg.selectedFiles()
            json_file = json_file[0]
            self.model.json_load(json_file)
            self.update_boxes()

    def save_json_settings(self):
        dlg = QtWidgets.QFileDialog(self, 'Choose save directory')
        json_filter = "JSON File (*.json)"
        # dlg.setNameFilter(json_filter)
        json_file = str()
        if dlg.exec_():
            json_file = dlg.getSaveFileName(self, filter=json_filter, caption="Save json file as:")
            # print(json_file[0])
            json_file = json_file[0]
            self.model.json_save(json_file)

    def plot_main(self):
        title = "placeholder"


def main():
    app = QtWidgets.QApplication(sys.argv)
    window1 = MainWindow()
    window1.show()
    app.exec_()


if __name__ == '__main__':
    main()
