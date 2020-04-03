#!/usr/bin/env python3

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import sys
import os
from MainWindow_2d import Ui_MainWindow
import faultslipMain
import json


class IOModel(QtCore.QAbstractListModel):
    def __init__(self, *args, inputs=None, outputs=None, **kwargs):
        super(IOModel, self).__init__(*args, **kwargs)
        self.inputs = inputs or dict()
        self.outputs = outputs or dict()

        # initialize defaults
        self.inputs['flag'] = 'mc'
        self.inputs['dip'] = 90
        self.inputs['dipunc'] = 10
        self.inputs['mu'] = 0.6
        self.inputs['mu_unc'] = 0.1
        self.inputs['pf'] = 75.0
        self.inputs['fail_percent'] = 50

    def json_load(self, infile):
        with open(infile) as load_file:
            j_data = json.load(load_file)
        self.inputs = j_data['input_data'][0]

    def json_save(self, outfile):
        outdict = dict()
        outdict['input_data'] = self.inputs
        with open(outfile, 'w') as dump_file:
            json.dump(outdict, dump_file, indent=4, sort_keys=True)

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.model = IOModel()

        # setup input boxes
        self.depth_mean_box.valueChanged.connect(self.depth_changed)
        self.shmax_mean_box.valueChanged.connect(self.shmax_mean_changed)
        self.shmax_std_box.valueChanged.connect(self.shmax_std_changed)
        self.shmin_mean_box.valueChanged.connect(self.shmin_mean_changed)
        self.shmin_std_box.valueChanged.connect(self.shmin_std_changed)
        self.sv_mean_box.valueChanged.connect(self.sv_mean_changed)
        self.sv_std_box.valueChanged.connect(self.sv_std_changed)
        self.shmax_az_mean_box.valueChanged.connect(self.shmax_az_mean_changed)
        self.shmin_az_mean_box.valueChanged.connect(self.shmin_az_mean_changed)
        self.shmax_az_std_box.valueChanged.connect(self.shmax_az_std_changed)
        self.shmin_az_std_box.valueChanged.connect(self.shmin_az_std_changed)
        self.sv_inc_unc_box.valueChanged.connect(self.sv_inc_unc_changed)
        self.dip_mean_box.valueChanged.connect(self.dip_mean_changed)
        self.dip_std_box.valueChanged.connect(self.dip_std_changed)
        self.mu_mean_box.valueChanged.connect(self.mu_mean_changed)
        self.mu_std_box.valueChanged.connect(self.mu_std_changed)
        self.max_pf_box.valueChanged.connect(self.max_pf_changed)
        self.fail_percentile_box.valueChanged.connect(self.fail_percentile_changed)
        self.det_model_yes.toggled.connect(self.det_model_enabled)
        self.prob_model_yes.toggled.connect(self.prob_model_enabled)

        # file dialog box
        self.file_browse_button.clicked.connect(self.shapefile_browse)
        self.shapefile_edit.textEdited.connect(self.shapefile_edit_text)

        # execute button
        self.executeButton.clicked.connect(self.execute_model)

        # menubar
        self.actionOpen_json_file.triggered.connect(self.open_json_settings)
        self.actionSave_json_settings.triggered.connect(self.save_json_settings)

    def depth_changed(self):
        self.model.inputs['depth'] = self.depth_mean_box.value()
        # print(self.model.inputs['depth'])

    def shmax_mean_changed(self):
        self.model.inputs['shmax'] = self.shmax_mean_box.value()

    def shmax_std_changed(self):
        self.model.inputs['shMunc'] = self.shmax_std_box.value()

    def shmin_mean_changed(self):
        self.model.inputs['shmin'] = self.shmin_mean_box.value()

    def shmin_std_changed(self):
        self.model.inputs['shmiunc'] = self.shmin_std_box.value()

    def sv_mean_changed(self):
        self.model.inputs['sv'] = self.sv_mean_box.value()

    def sv_std_changed(self):
        self.model.inputs['svunc'] = self.sv_std_box.value()

    def shmax_az_mean_changed(self):
        self.model.inputs['shmaxaz'] = self.shmax_az_mean_box.value()

    def shmin_az_mean_changed(self):
        self.model.inputs['shminaz'] = self.shmin_az_mean_box.value()

    def shmax_az_std_changed(self):
        self.model.inputs['azunc'] = self.shmax_az_std_box.value()
        if self.shmin_az_std_box.value() != self.shmax_az_std_box.value():
            self.shmin_az_std_box.setValue(self.shmax_az_std_box.value())

    def shmin_az_std_changed(self):
        self.model.inputs['azunc'] = self.shmin_az_std_box.value()
        if self.shmin_az_std_box.value() != self.shmax_az_std_box.value():
            self.shmax_az_std_box.setValue(self.shmin_az_std_box.value())

    def sv_inc_unc_changed(self):
        self.model.inputs['inc_unc'] = self.sv_inc_unc_box.value()

    def dip_mean_changed(self):
        self.model.inputs['dip'] = self.dip_mean_box.value()

    def dip_std_changed(self):
        self.model.inputs['dipunc'] = self.dip_std_box.value()

    def mu_mean_changed(self):
        self.model.inputs['mu'] = self.mu_mean_box.value()

    def mu_std_changed(self):
        self.model.inputs['mu_unc'] = self.mu_std_box.value()

    def max_pf_changed(self):
        self.model.inputs['pf'] = self.max_pf_box.value()

    def fail_percentile_changed(self):
        self.model.inputs['fail_percent'] = self.fail_percentile_box.value()

    def det_model_enabled(self):
        self.model.inputs['flag'] = 'det'
        self.prob_model_yes.setChecked(False)

    def prob_model_enabled(self):
        self.model.inputs['flag'] = 'mc'
        self.det_model_yes.setChecked(False)

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

    def execute_model(self):
        inputs = self.model.inputs
        infile = inputs['shapefile']
        type_flag = inputs['flag']
        results, bounds, plot_min, plot_max = faultslipMain.slip_tendency_2d(infile, inputs, type_flag)

    def update_boxes(self):
        self.depth_mean_box.setValue(self.model.inputs['depth'])
        self.shmax_mean_box.setValue(self.model.inputs['shmax'])
        self.shmax_std_box.setValue(self.model.inputs['shMunc'])
        self.shmin_mean_box.setValue(self.model.inputs['shmin'])
        self.shmin_std_box.setValue(self.model.inputs['shmiunc'])
        self.sv_mean_box.setValue(self.model.inputs['sv'])
        self.sv_std_box.setValue(self.model.inputs['svunc'])
        self.shmax_az_mean_box.setValue(self.model.inputs['shmaxaz'])
        self.shmin_az_mean_box.setValue(self.model.inputs['shminaz'])
        self.shmax_az_std_box.setValue(self.model.inputs['azunc'])
        self.sv_inc_unc_box.setValue(self.model.inputs['inc_unc'])
        self.dip_mean_box.setValue(self.model.inputs['dip'])
        self.dip_std_box.setValue(self.model.inputs['dipunc'])
        self.mu_mean_box.setValue(self.model.inputs['mu'])
        self.mu_std_box.setValue(self.model.inputs['mu_unc'])
        self.max_pf_box.setValue(self.model.inputs['pf'])
        self.fail_percentile_box.setValue(self.model.inputs['fail_percent'])

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



def main():
    app = QtWidgets.QApplication(sys.argv)
    window1 = MainWindow()
    window1.show()
    app.exec_()


if __name__ == '__main__':
    main()
