import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import default
import pickle
from classes import Cognit
from utility import show_connections, create_activations_history, record_activations_history, create_durability_history, record_durability_history
from sensors import Sensor_Pattern
from learning import learn
from PyQt5 import QtCore, QtWidgets
import time
from utility import connect, disconnect, change_sensors

class ConnectionsWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.layout = QtWidgets.QGridLayout()
        self.layout.setSpacing(20)

        ## Labels ##

        self.input_column_label = QtWidgets.QLabel()
        self.input_column_label.setText('Input')
        self.layout.addWidget(self.input_column_label, 0, 0, QtCore.Qt.AlignCenter)
        self.output_column_label = QtWidgets.QLabel()
        self.output_column_label.setText('Output')
        self.layout.addWidget(self.output_column_label, 0, 1, QtCore.Qt.AlignCenter)

        self.input_collections_label = QtWidgets.QLabel()
        self.input_collections_label.setText('Input connections')
        self.layout.addWidget(self.input_collections_label, 0, 2, QtCore.Qt.AlignCenter)
        self.output_collections_label = QtWidgets.QLabel()
        self.output_collections_label.setText('Output connections')
        self.layout.addWidget(self.output_collections_label, 0, 3, QtCore.Qt.AlignCenter)

        ## Combo box for layers ##
        self.input_layers_box = QtWidgets.QComboBox(self.centralwidget)
        self.output_layers_box = QtWidgets.QComboBox(self.centralwidget)
        self.layout.addWidget(self.input_layers_box, 1, 0)
        self.layout.addWidget(self.output_layers_box, 1, 1)
        self.return_list('input', 'layer')
        self.return_list('output', 'layer')
        self.input_layers_box.currentIndexChanged.connect(lambda state, level=0: self.box_changed(level, 'input'))
        self.output_layers_box.currentIndexChanged.connect(lambda state, level=0: self.box_changed(level, 'output'))

        ## Combo box for columns ##
        self.input_columns_box = QtWidgets.QComboBox(self.centralwidget)
        self.output_columns_box = QtWidgets.QComboBox(self.centralwidget)
        self.layout.addWidget(self.input_columns_box, 2, 0)
        self.layout.addWidget(self.output_columns_box, 2, 1)
        self.return_list('input', 'column')
        self.return_list('output', 'column')
        self.input_columns_box.currentIndexChanged.connect(lambda state, level=1: self.box_changed(level, 'input'))
        self.output_columns_box.currentIndexChanged.connect(lambda state, level=1: self.box_changed(level, 'output'))

        ## Combo box for neurons ##
        self.input_neurons_box = QtWidgets.QComboBox(self.centralwidget)
        self.output_neurons_box = QtWidgets.QComboBox(self.centralwidget)
        self.layout.addWidget(self.input_neurons_box, 3, 0)
        self.layout.addWidget(self.output_neurons_box, 3, 1)
        self.return_list('input', 'neuron')
        self.return_list('output', 'neuron')
        self.input_neurons_box.currentIndexChanged.connect(lambda state, level=2: self.box_changed(level, 'input'))
        self.output_neurons_box.currentIndexChanged.connect(lambda state, level=2: self.box_changed(level, 'output'))

        ## Combo box for dendrites ##
        self.input_dendrites_box = QtWidgets.QComboBox(self.centralwidget)
        self.output_dendrites_box = QtWidgets.QComboBox(self.centralwidget)
        self.layout.addWidget(self.input_dendrites_box, 4, 0)
        self.layout.addWidget(self.output_dendrites_box, 4, 1)
        self.return_list('input', 'dendrite')
        self.return_list('output', 'dendrite')

        ## Buttons to show connections ##
        self.input_show_button = QtWidgets.QPushButton(self.centralwidget)
        self.input_show_button.setText("Show")
        self.layout.addWidget(self.input_show_button, 5, 0)
        self.output_show_button = QtWidgets.QPushButton(self.centralwidget)
        self.output_show_button.setText("Show")
        self.layout.addWidget(self.output_show_button, 5, 1)
        self.input_show_button.clicked.connect(lambda state, in_out = 0: self.show_input_list(in_out))
        self.output_show_button.clicked.connect(lambda state, in_out = 1: self.show_input_list(in_out))

        ## List of input connections ##
        self.input_connections_list = QtWidgets.QListWidget(self.centralwidget)
        self.layout.addWidget(self.input_connections_list, 1, 2, 6, 1)

        ## List of output connections ##
        self.output_connections_list = QtWidgets.QListWidget(self.centralwidget)
        self.layout.addWidget(self.output_connections_list, 1, 3, 6, 1)

        ## Connect and disconnect button ##
        self.connect_button = QtWidgets.QPushButton()
        self.connect_button.setText("Connect")
        self.connect_button.clicked.connect(lambda state, connect_disconnect = True: self.connect_items(connect_disconnect))
        self.layout.addWidget(self.connect_button, 6, 0)
        self.disconnect_button = QtWidgets.QPushButton()
        self.disconnect_button.setText("Disconnect")
        self.disconnect_button.clicked.connect(lambda state, connect_disconnect=False: self.connect_items(connect_disconnect))
        self.layout.addWidget(self.disconnect_button, 6, 1)

        self.setLayout(self.layout)

    def connect_items(self, connect_disconnect):
        in_layer_index = self.input_layers_box.currentIndex()
        in_column_index = self.input_columns_box.currentIndex()
        in_neuron_index = self.input_neurons_box.currentIndex()
        in_dendrite_index = self.input_dendrites_box.currentIndex()
        out_layer_index = self.output_layers_box.currentIndex()
        out_column_index = self.output_columns_box.currentIndex()
        out_neuron_index = self.output_neurons_box.currentIndex()
        out_dendrite_index = self.output_dendrites_box.currentIndex()

        if in_neuron_index == 0:
            item1 = cognit.layer[in_layer_index].column[in_column_index].inhibition_neuron
        elif in_neuron_index == len(cognit.layer[in_layer_index].column[in_column_index].interneuron) + 1:
            item1 = cognit.layer[in_layer_index].column[in_column_index].output_neuron
        else:
            if len(cognit.layer[in_layer_index].column[in_column_index].interneuron) != 0:
                if in_dendrite_index == 0:
                    item1 = cognit.layer[in_layer_index].column[in_column_index].interneuron[in_neuron_index-1]
                else:
                    item1 = cognit.layer[in_layer_index].column[in_column_index].interneuron[in_neuron_index - 1].dendrites[in_dendrite_index - 1]

        if out_neuron_index == 0:
            item2 = cognit.layer[out_layer_index].column[out_column_index].inhibition_neuron
        elif out_neuron_index == len(cognit.layer[out_layer_index].column[out_column_index].interneuron) + 1:
            item2 = cognit.layer[out_layer_index].column[out_column_index].output_neuron
        else:
            if len(cognit.layer[out_layer_index].column[out_column_index].interneuron) != 0:
                if out_dendrite_index == 0:
                    item2 = cognit.layer[out_layer_index].column[out_column_index].interneuron[out_neuron_index-1]
                else:
                    item2 = cognit.layer[out_layer_index].column[out_column_index].interneuron[out_neuron_index - 1].dendrites[out_dendrite_index - 1]

        if connect_disconnect == True: connect(item1, item2)
        else: disconnect(item1, item2)
    def show_input_list(self, in_out):
        self.input_connections_list.clear()
        self.output_connections_list.clear()
        if in_out == 0:
            layer_index = self.input_layers_box.currentIndex()
            column_index = self.input_columns_box.currentIndex()
            neuron_index = self.input_neurons_box.currentIndex()
            dendrite_index = self.input_dendrites_box.currentIndex()
        elif in_out == 1:
            layer_index = self.output_layers_box.currentIndex()
            column_index = self.output_columns_box.currentIndex()
            neuron_index = self.output_neurons_box.currentIndex()
            dendrite_index = self.output_dendrites_box.currentIndex()

        if neuron_index == 0:
            item = cognit.layer[layer_index].column[column_index].inhibition_neuron
            for i in range(len(item.input_connections)):
                self.input_connections_list.addItem(item.input_connections[i].input.comment)

        elif neuron_index == len(cognit.layer[layer_index].column[column_index].interneuron) + 1:
            if len(cognit.layer[layer_index].column[column_index].output_neuron) != 0:
                for item in cognit.layer[layer_index].column[column_index].output_neuron:
                    for i in range(len(item.output_connections)):
                        self.output_connections_list.addItem(item.output_connections[i].output.comment)

        else:
            if len(cognit.layer[layer_index].column[column_index].interneuron) != 0:
                if dendrite_index == 0:
                    item = cognit.layer[layer_index].column[column_index].interneuron[neuron_index-1]
                    for k in range(len(item.input_connections)):
                        self.input_connections_list.addItem(item.input_connections[k].input.comment)
                    for l in range(len(item.output_connections)):
                        self.output_connections_list.addItem(item.output_connections[l].output.comment)
                else:
                    item = cognit.layer[layer_index].column[column_index].interneuron[neuron_index-1].dendrites[dendrite_index - 1]
                    for k in range(len(item.input_connections)):
                        self.input_connections_list.addItem(item.input_connections[k].input.comment)
    def return_list(self, in_out, level):
        to_show_list = []
        if level == 'layer':
            if len(cognit.layer) != 0:
                for layer in cognit.layer:
                    to_show_list = to_show_list + [layer.comment]
            if in_out == 'input':
                self.input_layers_box.clear()
                self.input_layers_box.addItems(to_show_list)
            elif in_out == 'output':
                self.output_layers_box.clear()
                self.output_layers_box.addItems(to_show_list)

        elif level == 'column':
            if in_out == 'input': layer_index = self.input_layers_box.currentIndex()
            elif in_out == 'output': layer_index = self.output_layers_box.currentIndex()
            else: layer_index = 0
            for column in cognit.layer[layer_index].column:
                to_show_list = to_show_list + [column.comment]
            if in_out == 'input':
                self.input_columns_box.clear()
                self.input_columns_box.addItems(to_show_list)
            elif in_out == 'output':
                self.output_columns_box.clear()
                self.output_columns_box.addItems(to_show_list)

        elif level == 'neuron':
            if in_out == 'input':
                layer_index = self.input_layers_box.currentIndex()
                column_index = self.input_columns_box.currentIndex()
            elif in_out == 'output':
                layer_index = self.output_layers_box.currentIndex()
                column_index = self.output_columns_box.currentIndex()
            else:
                layer_index = 0
                column_index = 0
            to_show_list = to_show_list + [
                cognit.layer[layer_index].column[column_index].inhibition_neuron.comment]
            if len(cognit.layer[layer_index].column[column_index].interneuron) != 0:
                for neuron in cognit.layer[layer_index].column[column_index].interneuron:
                    to_show_list = to_show_list + [neuron.comment]
            if len(cognit.layer[layer_index].column[column_index].output_neuron) != 0:
                for neuron in cognit.layer[layer_index].column[column_index].output_neuron:
                    to_show_list = to_show_list + [neuron.comment]
            if in_out == 'input':
                self.input_neurons_box.clear()
                self.input_neurons_box.addItems(to_show_list)
            elif in_out == 'output':
                self.output_neurons_box.clear()
                self.output_neurons_box.addItems(to_show_list)

        elif level == 'dendrite':
            if in_out == 'input':
                layer_index = self.input_layers_box.currentIndex()
                column_index = self.input_columns_box.currentIndex()
                neuron_index = self.input_neurons_box.currentIndex()
            elif in_out == 'output':
                layer_index = self.output_layers_box.currentIndex()
                column_index = self.output_columns_box.currentIndex()
                neuron_index = self.output_neurons_box.currentIndex()
            else:
                layer_index = 0
                column_index = 0
                neuron_index = 0
            to_show_list = ['body']
            if neuron_index != 0 and neuron_index != (len(cognit.layer[layer_index].column[column_index].interneuron) + 1) and neuron_index!= -1:
                if len(cognit.layer[layer_index].column[column_index].interneuron[neuron_index - 1].dendrites) != 0:
                    for dendrite in cognit.layer[layer_index].column[column_index].interneuron[neuron_index-1].dendrites:
                        to_show_list = to_show_list + [dendrite.comment]
            if in_out == 'input':
                self.input_dendrites_box.clear()
                self.input_dendrites_box.addItems(to_show_list)
            elif in_out == 'output':
                self.output_dendrites_box.clear()
                self.output_dendrites_box.addItems(to_show_list)

    def box_changed(self, level, in_out):
        if in_out == 'input':
            if level == 0:
                self.return_list('input','column')
                self.return_list('input','neuron')
                self.return_list('input','dendrite')
            if level == 1:
                self.return_list('input', 'neuron')
                self.return_list('input', 'dendrite')
            if level == 2:
                self.return_list('input', 'dendrite')

        if in_out == 'output':
            if level == 0:
                self.return_list('output','column')
                self.return_list('output','neuron')
                self.return_list('output','dendrite')
            if level == 1:
                self.return_list('output', 'neuron')
                self.return_list('output', 'dendrite')
            if level == 2:
                self.return_list('output', 'dendrite')

class SensorsWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.layout = QtWidgets.QGridLayout()
        self.layout.setSpacing(20)
        self.setFixedWidth(1500)
        self.setFixedHeight(300)
        self.sensor_label = QtWidgets.QLabel()
        self.sensor_label.setText('Sensor')
        self.layout.addWidget(self.sensor_label, 0, 0, QtCore.Qt.AlignCenter)

        ## Show array button ##
        self.show_button = QtWidgets.QPushButton(self.centralwidget)
        self.show_button.setGeometry(QtCore.QRect(80, 20, 161, 61))
        self.show_button.setText("Show array")
        self.show_button.setObjectName("Show attay")
        self.layout.addWidget(self.show_button, 0, 1)
        self.show_button.clicked.connect(self.show_sensors)

        ## Show value button ##
        self.show_value_button = QtWidgets.QPushButton(self.centralwidget)
        self.show_value_button.setGeometry(QtCore.QRect(80, 20, 161, 61))
        self.show_value_button.setText("Show value")
        self.show_value_button.setObjectName("Show value")
        self.layout.addWidget(self.show_value_button, 3, 1)
        self.show_value_button.clicked.connect(self.show_value)
    #
        ## Text Area for sensor activations ##
        self.show_area = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.show_area.setGeometry(QtCore.QRect(820, 10, 750, 900))
        # self.show_area.resize(700,700)
        self.layout.addWidget(self.show_area,0,3,6,1)
    #
        ## Sensor array combo box ##
        self.sensor_array_box = QtWidgets.QComboBox(self.centralwidget)
        self.layout.addWidget(self.sensor_array_box, 1, 0)
        self.return_sensor('layer')
        self.sensor_array_box.currentIndexChanged.connect(lambda state, type='layer': self.box_changed(type))

        ## Sensor position combo box ##
        self.sensor_position_x_box = QtWidgets.QComboBox(self.centralwidget)
        self.layout.addWidget(self.sensor_position_x_box, 2, 0)
        self.return_sensor('position_x')
        # self.sensor_array_box.currentIndexChanged.connect(lambda state, level=0: self.box_changed(level, '1D'))
    #
        ## Sensor position combo box ##
        self.sensor_position_y_box = QtWidgets.QComboBox(self.centralwidget)
        self.layout.addWidget(self.sensor_position_y_box, 3, 0)
        self.return_sensor('position_y')

        ## Sensor value box ##
        self.sensor_value = QtWidgets.QLineEdit(self.centralwidget)
        self.layout.addWidget(self.sensor_value, 1, 1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        self.sensor_value.setSizePolicy(sizePolicy)

        ## Sensor change value button ##
        self.change_button = QtWidgets.QPushButton(self.centralwidget)
        self.change_button.setGeometry(QtCore.QRect(80, 20, 161, 61))
        self.change_button.setText("Change")
        self.layout.addWidget(self.change_button, 2, 1)
        self.change_button.clicked.connect(self.change_value)
        self.setLayout(self.layout)
    def change_value(self):
        new_value = float(self.sensor_value.text())
        array_index = self.sensor_array_box.currentIndex()
        if cognit.sensor_layer[array_index].y_size == 1:
            y_index = 0
            x_index = self.sensor_position_y_box.currentIndex()
        else:
            y_index = self.sensor_position_y_box.currentIndex()
            x_index = self.sensor_position_x_box.currentIndex()
        signal_array_list[array_index][y_index][x_index] = new_value
    def show_value(self):
        self.sensor_value.clear()
        array_index = self.sensor_array_box.currentIndex()
        if cognit.sensor_layer[array_index].y_size == 1:
            y_index = 0
            x_index = self.sensor_position_y_box.currentIndex()
        else:
            y_index = self.sensor_position_y_box.currentIndex()
            x_index = self.sensor_position_x_box.currentIndex()

        self.sensor_value.setText(str(signal_array_list[array_index][y_index][x_index]))
    def show_sensors(self):
        self.show_area.clear()
        sensor_array = signal_array_list[self.sensor_array_box.currentIndex()]
        y_size = len(sensor_array)
        x_size = len(sensor_array[0])
        line = ''

        for j in range(y_size):
            for i in range(x_size):
                line = line + "{:.2f}".format(sensor_array[j][i]) + '\t'

            line = line + '\n\n'

        self.show_area.insertPlainText(line)

    def return_sensor(self, type):
        to_show_list = []
        if type == 'layer':
            for sensor_layer in cognit.sensor_layer:
                to_show_list = to_show_list + [sensor_layer.comment]

            self.sensor_array_box.clear()
            self.sensor_array_box.addItems(to_show_list)

        if type == 'position_x':
            to_show_list = []
            array_index = self.sensor_array_box.currentIndex()
            if cognit.sensor_layer[array_index].y_size == 1:
                for i in range(cognit.sensor_layer[array_index].x_size):
                    to_show_list = to_show_list + [str(i)]
            else:
                for i in range(cognit.sensor_layer[array_index].y_size):
                    to_show_list = to_show_list + [str(i)]

            self.sensor_position_x_box.clear()
            self.sensor_position_x_box.addItems(to_show_list)

        if type == 'position_y':
            to_show_list = []
            array_index = self.sensor_array_box.currentIndex()
            if cognit.sensor_layer[array_index].y_size == 1:
                to_show_list = ['']
            else:
                for i in range(cognit.sensor_layer[array_index].x_size):
                    to_show_list = to_show_list + [str(i)]

            self.sensor_position_y_box.clear()
            self.sensor_position_y_box.addItems(to_show_list)
    def box_changed(self, type):
        if type == 'layer':
            self.return_sensor('position_x')
            self.return_sensor('position_y')

class UiMainWindow(object):    ## Main window, partially QtDesigner generated ##
    def setupUi(self, MainWindow):
        self.reward_value = 0.0
        ##  Setting up main window ##
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1600, 950)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        ## Setting up frame for matplotlib plot ##
        self.canvas_frame = QtWidgets.QFrame(self.centralwidget)
        self.canvas_frame.setGeometry(QtCore.QRect(820, 10, 750, 900))
        self.canvas_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.canvas_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.canvas_frame.setObjectName("Canvas frame")

        ## Button for single run ##
        self.single_run_button = QtWidgets.QPushButton(self.centralwidget)
        self.single_run_button.setGeometry(QtCore.QRect(80, 20, 161, 61))
        self.single_run_button.setText("Single Run")
        self.single_run_button.setObjectName("Single Run")
        self.single_run_button.clicked.connect(self.single_run)

        ## Button for update plot without feeding forward ##
        self.update_button = QtWidgets.QPushButton(self.centralwidget)
        self.update_button.setGeometry(QtCore.QRect(255,20,161,61))
        self.update_button.setText("Update")
        self.update_button.clicked.connect(self.update_plot)

        ## Button for several runs with input for number of runs ##
        self.many_runs_button = QtWidgets.QPushButton(self.centralwidget)
        self.many_runs_button.setGeometry(QtCore.QRect(430, 20, 161, 61))
        self.many_runs_button.setObjectName("Several Runs")
        self.many_runs_button.setText("Several Runs")
        self.many_runs_lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.many_runs_lineEdit.setGeometry(QtCore.QRect(275, 100, 51, 31))
        self.many_runs_lineEdit.setObjectName("Runs number")
        self.many_runs_lineEdit.setText("1")
        self.many_runs_button.clicked.connect(self.many_runs)
        self.many_runs_title = QtWidgets.QLabel(self.centralwidget)
        self.many_runs_title.setGeometry(342,65, 80,100)
        self.many_runs_title.setText('runs')
        MainWindow.setCentralWidget(self.centralwidget)

        ## Button for stopping the calcualtion process ##
        self.stop_button = QtWidgets.QPushButton(self.centralwidget)
        self.stop_button.setGeometry(QtCore.QRect(605, 20, 161, 61))
        self.stop_button.setObjectName("Stop")
        self.stop_button.setText("Stop")
        self.stop_button.setCheckable(True)
        self.stop_button.setChecked(False)
        self.stop_button.clicked.connect(self.stop)

        ## Box for turning animation on and off ##
        self.check_animate_title = QtWidgets.QLabel(self.centralwidget)
        self.check_animate_title.setGeometry(130,73,80,80)
        self.check_animate_title.setText('Animate')
        self.check_animate = QtWidgets.QCheckBox(self.centralwidget)
        self.check_animate.setGeometry(100, 105, 20, 20)
        self.check_animate.setChecked(False)

        ## Box to allow movement ##
        self.check_move_title = QtWidgets.QLabel(self.centralwidget)
        self.check_move_title.setGeometry(480,73,120,80)
        self.check_move_title.setText('Allow movement')
        self.check_move = QtWidgets.QCheckBox(self.centralwidget)
        self.check_move.setGeometry(450, 105, 20, 20)
        self.check_move.setChecked(False)

        ## Button for connections ##
        self.connections_button = QtWidgets.QPushButton(self.centralwidget)
        self.connections_button.setGeometry(QtCore.QRect(80, 200, 161, 61))
        self.connections_button.setText('Connections')
        self.connections_button.clicked.connect(self.show_connections)

        ## Button for sensors ##
        self.sensors_button = QtWidgets.QPushButton(self.centralwidget)
        self.sensors_button.setGeometry(QtCore.QRect(255, 200, 161, 61))
        self.sensors_button.setText('Sensors')
        self.sensors_button.clicked.connect(self.show_sensors)

        ## Log settings ##
        self.log_mode_label = QtWidgets.QLabel(self.centralwidget)
        self.log_mode_label.setGeometry(80,360, 100,30)
        self.log_mode_label.setText('Log mode:')
        self.log_mode = QtWidgets.QComboBox(self.centralwidget)
        self.log_mode.setGeometry(80, 400, 160, 30)
        self.log_mode.addItems(['Interneuron info'])
        self.log_mode_button = QtWidgets.QPushButton(self.centralwidget)
        self.log_mode_button.setGeometry(80, 440, 161, 61)
        self.log_mode_button.setText('Set log mode')
        self.log_mode_button.clicked.connect(self.log_mode_button_clicked)


        ## Log select for interneurons ##
        self.log_select_layer = QtWidgets.QComboBox(self.centralwidget)
        self.log_select_layer.setGeometry(250,400,160,30)
        self.log_select_layer.currentIndexChanged.connect(lambda state: self.log_layer_changed)
        self.log_select_column = QtWidgets.QComboBox(self.centralwidget)
        self.log_select_column.setGeometry(250,440,160,30)
        self.log_select_neuron = QtWidgets.QComboBox(self.centralwidget)
        self.log_select_neuron.setGeometry(250, 480, 160,30)
        self.log_select_button = QtWidgets.QPushButton(self.centralwidget)
        self.log_select_button.setGeometry(80,510, 161,61)
        self.log_select_button.setText('Set neuron choice')

        # if self.log_mode.currentText() == 'Interneuron info':


        ## Log for events ##
        # self.log_title = QtWidgets.QLabel(self.centralwidget)
        # self.log_title.setGeometry(85,545,80,80)
        # self.log_title.setText('Log')
        self.log_window = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.log_window.setGeometry(QtCore.QRect(80, 600, 680, 300))

        ## Setting up menus
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1600, 31))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        ## Setting up status bars
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout = QtWidgets.QVBoxLayout(self.canvas_frame)
        self.layout.addWidget(self.canvas)

    def log_layer_changed(self):

    def log_mode_button_clicked(self):
        if self.log_mode.currentText() == 'Interneuron info':
            self.log_window.setPlaceholderText('Log mode: Solo interneuron information\n')
            self.log_select_layer.clear()
            self.log_select_column.clear()
            self.log_select_neuron.clear()
            to_show_list = []
            for layer in cognit.layer:
                to_show_list = to_show_list + [layer.comment]
            self.log_select_layer.addItems(to_show_list)
            to_show_list = []
            for column in cognit.layer[0].column:
                to_show_list = to_show_list + [column.comment]
            self.log_select_column.addItems(to_show_list)
            to_show_list = []
            for neuron in cognit.layer[0].column[0].interneuron:
                to_show_list = to_show_list + [neuron.comment]
            self.log_select_neuron.addItems(to_show_list)


    def show_sensors(self):
        self.window = SensorsWindow()
        self.window.show()
    def show_connections(self):
        self.window = ConnectionsWindow()
        self.window.show()
    def stop(self):
        self.stop_button.setChecked(True)
    def update_plot(self):
        self.plot()

    def show_in_log(self, connection):
        self.log_window.setPlaceholderText(connection.input.comment + 'to' + connection.output.comment)
    def single_run(self):
        if self.check_animate.isChecked():
            cognit.clean_up()
            cognit.feed_sensors(signal_array_list)
            self.plot(column = None)
            time.sleep(0.2)
            if len(cognit.activation_list) != 0:
                i = 0
                while i < len(cognit.activation_list):
                    column = cognit.activation_list[i]
                    column.feed_forward()
                    self.plot(column)
                    time.sleep(0.2)
                    i += 1
            else: self.plot()
            signal_array_list[2][0][0] += default.grow_impatience
            if self.check_move.isChecked():
                self.move()

        else:
            cognit.clean_up()
            cognit.feed_sensors(signal_array_list)
            cognit.feed_forward()

            if self.check_move.isChecked():
                self.move()

            if len(cognit.new_connections) !=0:
                for connection in cognit.new_connections:
                    self.show_in_log(connection)


            self.plot(column = None)
            time.sleep(0.2)
            signal_array_list[2][0][0] +=  default.grow_impatience

    def many_runs(self):
        number_of_times = int(self.many_runs_lineEdit.text())
        for i in range(number_of_times):
            self.single_run()
            if self.stop_button.isChecked(): break
        self.stop_button.setChecked(False)

    def move(self):
        for interneuron in cognit.layer[1].column[0].interneuron:
            if interneuron.activation != 0:
                signal_array_list[1][0][0] = 0.0
                if interneuron.comment == 'up':
                    signal_array_list[0] = change_sensors(signal_array_list[0], 'labyrinth_position', 'up')
                elif interneuron.comment == 'down':
                    signal_array_list[0] = change_sensors(signal_array_list[0], 'labyrinth_position', 'down')
                elif interneuron.comment == 'left':
                    signal_array_list[0] = change_sensors(signal_array_list[0], 'labyrinth_position', 'left')
                elif interneuron.comment == 'right':
                    signal_array_list[0] = change_sensors(signal_array_list[0], 'labyrinth_position', 'right')
                signal_array_list[2][0][0] = 0.0

    def plot(self, column = None):
        self.figure.clear()
        axis = self.figure.add_axes((0, 0, 1, 1))
        axis.set_aspect('equal')
        self.x_limit = 75
        self.y_limit = 90
        plt.xlim([0, self.x_limit])
        plt.ylim([0, self.y_limit])

        sensor_array_list_size = len(cognit.sensor_layer)
        sensor_window_y_size = 30
        sensor_window_x_size = self.x_limit / sensor_array_list_size
        columns_window_y_size = self.y_limit - sensor_window_y_size

        ## Horizontal border for sensors ##
        draw_line = plt.Line2D((0, self.x_limit),
                               (sensor_window_y_size, sensor_window_y_size),
                               lw=1.0,
                               ls='--',
                               color='blue')
        axis.add_artist(draw_line)

        ## Horizontal border for columns ##
        draw_line = plt.Line2D((0, self.x_limit),
                               (columns_window_y_size, columns_window_y_size),
                               lw=1.0,
                               ls='--',
                               color='blue')
        axis.add_artist(draw_line)

        for i in range(sensor_array_list_size):
            ## Vertical border for sensors ##
            draw_line = plt.Line2D((sensor_window_x_size * i, sensor_window_x_size * i),
                                   (0, sensor_window_y_size),
                                   lw=1.0,
                                   ls = '--',
                                   color='blue')
            axis.add_artist(draw_line)

            layer_x_size = cognit.sensor_layer[i].x_size
            if cognit.sensor_layer[i].y_size == None:
                layer_y_size = 1
            else: layer_y_size = cognit.sensor_layer[i].y_size
            radius = min((sensor_window_y_size /  layer_y_size / 4), (sensor_window_x_size / layer_x_size / 4))
            x_step = sensor_window_x_size / layer_x_size
            y_step = sensor_window_y_size / layer_y_size
            for x_index in range(layer_x_size):
                for y_index in range(layer_y_size):

                    x_position = x_step * (0.5 + x_index) + i * sensor_window_x_size
                    y_position = y_step * (0.5 + y_index)
                    if layer_y_size == 1: current_sensor = cognit.sensor_layer[i].sensor[x_index]
                    else:
                        for sensor in cognit.sensor_layer[i].sensor:
                            if sensor.position == [y_index, x_index]:
                                current_sensor = sensor
                                break
                            else: current_sensor = None

                    if current_sensor != None:
                        if current_sensor.activation != 0:
                            ## Sensor activation ##
                            draw_circle = plt.Circle((x_position, y_position),
                                                     radius * current_sensor.activation,
                                                     fill=True,
                                                     color='red')
                            axis.add_artist(draw_circle)
                    ## Sensor border ##
                    draw_circle = plt.Circle((x_position, y_position),
                                              radius,
                                              fill=False,
                                              color='black')
                    axis.add_artist(draw_circle)

        ## Drawing columns ##
        for i in range(len(cognit.layer)):
            layer_x_size = cognit.layer[i].x_size
            layer_y_size = cognit.layer[i].y_size
            rec_size = columns_window_y_size / len(cognit.layer)
            radius = min((self.x_limit / layer_x_size / 4), (rec_size / layer_y_size / 4))
            x_step = self.x_limit / layer_x_size
            y_step = rec_size / (layer_y_size + 1)

            for current_column in cognit.layer[i].column:
                x_position = x_step * (0.5 + current_column.position[1])
                y_position = sensor_window_y_size + rec_size * i + y_step * (current_column.position[0] + 1)
            #
                ## Activation of columns ##
                activation = 0.0
                for neuron in current_column.interneuron:
                    activation = max(activation, neuron.activation)
                draw_rectangle = plt.Rectangle(((x_position - radius / 2), (y_position - radius / 2)),
                                               height=radius * activation,
                                               width=radius * activation,
                                               fill=True,
                                               color='red')
                axis.add_artist(draw_rectangle)

                ## Border of columns ##
                draw_rectangle = plt.Rectangle((x_position - radius / 2, y_position - radius / 2),
                                               height=radius,
                                               width=radius,
                                               fill=False,
                                               color='black')
                axis.add_artist(draw_rectangle)

                if column != None:
                    if current_column == column:
                        ## Outline activated column ##
                        draw_rectangle = plt.Rectangle(((x_position - radius / 2) - 1, (y_position - radius / 2) - 1),
                                                       height=radius + 2,
                                                       width=radius + 2,
                                                       linewidth = 2.0,
                                                       fill=False,
                                                       color='blue')
                        axis.add_artist(draw_rectangle)

                    ## Outline for output activations ##
                    for neuron in column.output_neuron:
                        if len(neuron.output_connections) != 0:
                            for connection in neuron.output_connections:
                                if connection.output.column == current_column:
                                    draw_rectangle = plt.Rectangle(
                                        ((x_position - radius / 2) - 1, (y_position - radius / 2) - 1),
                                        height=radius + 2,
                                        width=radius + 2,
                                        linewidth=2.0,
                                        linestyle = '--',
                                        fill=False,
                                        color='green')
                                    axis.add_artist(draw_rectangle)

        self.canvas.draw()
        self.canvas.flush_events()

if __name__ == "__main__":
    global cognit
    global signal_array_list

    with open('current_cognit.pickle', 'rb') as f:
        cognit = pickle.load(f)
    # cognit.column[3].interneuron[0].threshold = 0.5
    signal_array_list = [[[0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 1.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0]], [[0.0]], [[0.0]]]
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = UiMainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
