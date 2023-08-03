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
from utility import connect, disconnect

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
            item = cognit.layer[layer_index].column[column_index].output_neuron
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
            to_show_list = to_show_list + [
                cognit.layer[layer_index].column[column_index].output_neuron.comment]
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

        ## Labels ##
        self.sensor_array_label = QtWidgets.QLabel()
        self.sensor_array_label.setText('Sensor array')
        self.layout.addWidget(self.sensor_array_label, 0, 0, QtCore.Qt.AlignCenter)

        self.sensor_1D_label = QtWidgets.QLabel()
        self.sensor_1D_label.setText('1D')
        self.layout.addWidget(self.sensor_1D_label, 1, 0, QtCore.Qt.AlignCenter)

        self.sensor_2D_label = QtWidgets.QLabel()
        self.sensor_2D_label.setText('2D')
        self.layout.addWidget(self.sensor_2D_label, 1, 1, QtCore.Qt.AlignCenter)

        ## Show button ##
        self.show_button = QtWidgets.QPushButton(self.centralwidget)
        self.show_button.setGeometry(QtCore.QRect(80, 20, 161, 61))
        self.show_button.setText("Show")
        self.show_button.setObjectName("Show")
        self.layout.addWidget(self.show_button,0,2)
        self.show_button.clicked.connect(self.show_sensors)

        ## Text Area for sensor activations ##
        self.show_area = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.show_area.setGeometry(QtCore.QRect(820, 10, 750, 900))
        # self.show_area.resize(700,700)
        self.layout.addWidget(self.show_area,0,3,6,1)

        ## Sensor array combo box ##
        self.sensor_array_box = QtWidgets.QComboBox(self.centralwidget)
        self.layout.addWidget(self.sensor_array_box, 0, 1)
        self.return_sensor('layer','to_show')
        # self.sensor_array_box.currentIndexChanged.connect(lambda state, level=0: self.box_changed(level, '1D'))

        ## 1D sensor pick combo box ##
        self.sensor_1d_box = QtWidgets.QComboBox(self.centralwidget)
        self.layout.addWidget(self.sensor_1d_box, 2, 0)
        self.return_sensor('layer','to_1D')
        self.sensor_1d_box.currentIndexChanged.connect(lambda state, level = 0: self.box_1d_changed(level))
        # self.output_columns_box.currentIndexChanged.connect(lambda state, level=1: self.box_changed(level, 'output'))

        self.sensor_1d_box_index = QtWidgets.QComboBox(self.centralwidget)
        self.layout.addWidget(self.sensor_1d_box_index, 3, 0)
        self.return_sensor('index', 'to_1D')
        self.sensor_1d_box_index.currentIndexChanged.connect(lambda state, level = 1: self.box_1d_changed(level))

        self.sensor_1d_value = QtWidgets.QLineEdit(self.centralwidget)
        self.layout.addWidget(self.sensor_1d_value,4,0)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        self.sensor_1d_value.setSizePolicy(sizePolicy)
        self.return_sensor_activation('1D')
        self.sensor_1d_value.returnPressed.connect(self.sensor_1d_changed)

        self.setLayout(self.layout)

    def box_1d_changed(self, level):
        if level == 0:
            self.return_sensor('index', 'to_1D')
        current_sensor_array_index = self.sensor_1d_box.currentIndex()
        current_sensor_index = self.sensor_1d_box_index.currentIndex()
        current_value = signal_array_list[current_sensor_array_index][current_sensor_index]
        self.sensor_1d_value.setText(str(current_value))


    def sensor_1d_changed(self):
        # current_sensor_array_index = self.sensor_1d_box.currentIndex()
        current_sensor_array_text = self.sensor_1d_box.currentText()
        current_sensor_index = self.sensor_1d_box_index.currentIndex()
        current_value = float(self.sensor_1d_value.text())
        for sensor_layer in cognit.sensor_layer:
            if sensor_layer.comment == current_sensor_array_text:
                cognit.sensor[current_sensor_array_index].sensor[current_sensor_index].activation = current_value
        signal_array_list[current_sensor_array_index][current_sensor_index] = current_value
        self.sensor_array_box.setCurrentIndex(current_sensor_array_index)
        self.show_sensors()
    def return_sensor_activation(self, type):
        if type == '1D':
            layer_index = self.sensor_1d_box.currentIndex()
            index = self.sensor_1d_box_index.currentIndex()
            value = signal_array_list[layer_index][index]
            self.sensor_1d_value.setText(str(value))
    def box_changed(self, level, where):
        if where == '1D':
            if level == 0:
                self.return_sensor('index','to_1D')
    def show_sensors(self):
        self.show_area.clear()
        sensor_index = self.sensor_array_box.currentIndex()
        sensor_layer = signal_array_list[sensor_index]
        line = ''
        size = len(sensor_layer)
        for i in range(size):
            line = line + '  ' + str(i) + '\t'
        line = line + '\n\n'
        for i in range(size):
            line = line + "{:.2f}".format(sensor_layer[i]) + '\t'

        self.show_area.insertPlainText(line)

    def return_sensor(self, type, where_to):
        to_show_list = []
        if type == 'layer':
            for sensor_layer in cognit.sensor_layer:
                if where_to == 'to_1D':
                    if sensor_layer.check_1D_2D == '1D':
                        to_show_list = to_show_list + [sensor_layer.comment]
                elif where_to == 'to_2D':
                    if sensor_layer.check_1D_2D == '2D':
                        to_show_list = to_show_list + [sensor_layer.comment]

            if where_to == 'to_show':
                self.sensor_array_box.clear()
                self.sensor_array_box.addItems(to_show_list)

            elif where_to == 'to_1D' :
                self.sensor_1d_box.clear()
                self.sensor_1d_box.addItems(to_show_list)

        if type == 'index':
            current_index = 0
            if where_to == 'to_1D':
                comment = self.sensor_1d_box.currentText()
                for sensor_layer in cognit.sensor_layer:
                    if sensor_layer.comment == comment:
                        for i in range(cognit.sensor_layer[current_index].x_size):
                            self.sensor_1d_box_index.addItems([str(i)])

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
        self.check_animate.setChecked(True)

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

        else:
            cognit.clean_up()
            cognit.feed_sensors(signal_array_list)
            if len(cognit.activation_list) != 0:
                i = 0
                while i < len(cognit.activation_list):
                    column = cognit.activation_list[i]
                    column.feed_forward()
                    i += 1

            self.plot(column = None)
            time.sleep(0.2)

    def many_runs(self):
        number_of_times = int(self.many_runs_lineEdit.text())
        for i in range(number_of_times):
            self.single_run()
            if self.stop_button.isChecked(): break
        self.stop_button.setChecked(False)

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
                    else: current_sensor = cognit.sensor_layer[i].sensor[y_index][x_index]

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
    signal_array_list = [[[0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.5, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0]], [0.5]]
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = UiMainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
