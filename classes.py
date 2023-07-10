import default
import numpy as np

class Cognit():
    def __init__(self):
        self.activation_list = []
        self.column = []
        self.layer = []
        self.sensor = []

    def feed_sensors(self,signal_array_list):
        self.activation_list = []

        ## Feed sensors and add activated columns to column activation list ##
        if len(signal_array_list) == len(self.sensor):
            for i in range(len(signal_array_list)):
                if len(signal_array_list[i]) == len(self.sensor[i].sensor):
                    if self.sensor[i].check_1D_2D == '1D':
                        for j in range(len(signal_array_list[i])):
                            if signal_array_list[i][j] != 0:
                                self.sensor[i].sensor[j].activation = signal_array_list[i][j]
                                if len(self.sensor[i].sensor[j].output_connections) != 0:
                                    for connection in self.sensor[i].sensor[j].output_connections:
                                        if connection.output.column not in self.activation_list:
                                            self.activation_list = self.activation_list + [
                                                connection.output.column]
                                        connection.output.stored_activation += signal_array_list[i][j]
                                        connection.output.stored_count += 1
                                        connection.output.stored_max = max(connection.output.stored_max, signal_array_list[i][j])
                else:
                    print("Signal and sensor sizes don't match")
        else:
            print("Signal and sensor arrays don't match")

    ## Feed forwarding sensors and columns in cognit ##
    def feed_forward(self, signal_array_list):
        # self.feed_sensors(signal_array_list)
        ## Feed forward columns in acctivation list ##
        if len(self.activation_list) != 0:
            i = 0
            while i < len(self.activation_list):
                self.activation_list[i].feed_forward()
                i += 1
    #
    # ## Creating sensor and adding it to list ##
    # def create_sensor_array(self, size, comment = ''):
    #     sensor_array = []
    #     for i in range(size):
    #         sensor = Sensor(comment + '_' + str(i), self)
    #         sensor_array = sensor_array + [sensor]
    #     self.sensor_array_list = self.sensor_array_list + [sensor_array]

    ## Adding layer of equal columns to cognit ##
    def create_column_layer(self, comment, n_columns, n_interneuron, n_dendrites):
        layer = Layer(cognit = self, n_columns = n_columns, comment = comment, n_interneuron = n_interneuron, n_dendrites = n_dendrites)
        self.layer = self.layer + [layer]
        for column in layer.column:
            self.column = self.column + [column]

    def create_sensor_layer(self, comment, check_1D_2D, x_size, y_size = None):
        sensor_layer = Sensor_Layer(cognit = self, comment = comment, check_1D_2D=check_1D_2D, x_size=x_size, y_size=y_size)
        self.sensor = self.sensor + [sensor_layer]

    ## Zeroing activations after feed forward ##
    def clean_up(self):
        for column in self.activation_list:
            column.clean_up()

class Sensor_Layer():
    def __init__(self, cognit, comment, check_1D_2D, x_size, y_size = None):
        self.sensor = []
        self.cognit = cognit
        self.comment = comment
        self.check_1D_2D = check_1D_2D
        self.x_size = x_size
        self.y_size = y_size
        if check_1D_2D == '1D':
            for i in range(x_size):
                sensor = Sensor(cognit = cognit, comment = comment + '_' + str(i))
                self.sensor = self.sensor + [sensor]

        if check_1D_2D == '2D':
            for i in range(y_size):
                sensor_line = []
                for j in range(x_size):
                    sensor = Sensor(cognit = cognit, comment = comment)
                    sensor_line = sensor_line + [sensor]
                self.sensor = self.sensor + [sensor_line]

class Layer():
    def __init__(self, cognit, comment, n_columns, n_interneuron, n_dendrites):
        self.column = []
        self.cognit = cognit
        self.comment = comment
        for i in range(n_columns):
            column = Column(cognit, layer = self, interneuron_number = n_interneuron, dendrites_number = n_dendrites, comment = comment + '_' + str(i))
            self.column = self.column + [column]

class Column():
    def __init__(self, cognit, layer, interneuron_number, dendrites_number, comment, output_number = 1):

        self.cognit = cognit
        self.layer = layer
        self.comment = comment
        self.inhibition_neuron = Inhibition_Neuron(column = self)
        self.inhibition_neuron.comment = self.comment + '_block'
        self.output_neuron = []
        for i in range(output_number):
            neuron = Output_Neuron(column = self)
            neuron.comment = self.comment + '_out' + str(i)
            self.output_neuron = self.output_neuron + [neuron]
        self.interneuron = []
        for i in range(interneuron_number):
            neuron = Interneuron(dendrites_number = dendrites_number, column = self)
            neuron.comment = self.comment + '_i' + str(i)
            for j in range(len(neuron.dendrites)):
                neuron.dendrites[j].comment = neuron.comment + '_d' + str(j)
                neuron.dendrites[j].column = self
            self.interneuron = self.interneuron + [neuron]

        self.body_receptive_field = []
        self.dendrite_receptive_field = []

    ## Feed forward inside the column (winner interneuron takes all based on activation function) ##
    def feed_forward(self):

    ## Inhibition of interneuron: preinhibition + activation function calculation + inhibition ##

        ## Preinhibition ##
        if self.inhibition_neuron.stored_count != 0:
            #####################################
            ## Activation of inhibition neuron ##
            self.inhibition_neuron.activation = self.inhibition_neuron.stored_activation * 0.1
            #####################################
            #####################################
            self.inhibition_neuron.stored_activation = 0.0
            self.inhibition_neuron.stored_max = 0.0
            self.inhibition_neuron.stored_count = 0

        if self.inhibition_neuron.activation != 0:
            for neuron in self.interneuron:
                neuron.stored_activation -= self.inhibition_neuron.activation
                neuron.stored_activation = max(neuron.stored_activation, 0.0)

        ## Calculating activation_function ##
        for neuron in self.interneuron:
            # Activating dendrites #
            neuron.prime = 0.0
            for dendrite in neuron.dendrites:
                if len(dendrite.input_connections) != 0:
                    if dendrite.stored_count != 0:
                        ####################################
                        ## Activation of dendrites #########
                        dendrite.activation = dendrite.stored_activation
                        ####################################
                        ####################################
                        dendrite.stored_activation = 0.0
                        dendrite.stored_max = 0.0
                        dendrite.stored_count = 0
                    ## setting threshold as medium activation required to fire ##
                    neuron.prime = max(neuron.prime, dendrite.activation)
                    dendrite.activation = 0.0
                    # record_connections(dendrite, record_inputs=True, record_outputs=False)

            if neuron.stored_count != 0:
                # neuron.activation = neuron.stored_activation / neuron.stored_count
                ############################################
                ## Activation of interneurons ##
                neuron.activation = neuron.stored_max
                ############################################
                ############################################
                neuron.stored_activation = 0.0
                neuron.stored_max = 0.0
                neuron.stored_count = 0.0

        ## Inhibition based on activation function ##
        neuron_max = None
        activation_max = 0.0
        for neuron in self.interneuron:
            if neuron.activation != 0 and ((neuron.activation + neuron.prime) > activation_max):
                neuron_max = neuron
        if neuron_max != None:
            for neuron in self.interneuron:
                if neuron != neuron_max: neuron.activation = 0.0

        ## Calculating activation of interneuron after inhibition ##
        for neuron in self.interneuron:
            if neuron.activation > neuron.neuron_threshold:
                neuron.trace_activation = neuron.activation
                for connection in neuron.output_connections:
                    connection.output.stored_activation += neuron.activation + 0.01 * np.random.randn()
                    connection.output.stored_max = max(connection.output.stored_max, neuron.activation)
                    connection.output.stored_count += 1
                    if connection.output.column not in self.cognit.activation_list:
                        self.cognit.activation_list = self.cognit.activation_list + [connection.output.column]
            neuron.activation = 0.0

        ## Calculating activation of output neuron ##
        for neuron in self.output_neuron:
            if neuron.stored_activation > neuron.neuron_threshold:
                neuron.activation = neuron.stored_activation
                for connection in neuron.output_connections:
                    connection.output.stored_activation += neuron.activation + 0.01 * np.random.randn()
                    connection.output.stored_max = max(connection.output.stored_max, neuron.activation)
                    connection.output.stored_count += 1
                    if connection.output.column not in self.cognit.activation_list:
                        self.cognit.activation_list = self.cognit.activation_list + [connection.output.column]

    ## Cleaning up activations inside column after feed forward ##
    def clean_up(self):
        self.inhibition_neuron.activation = 0.0
        for neuron in self.interneuron:
            neuron.activation = 0.0
        for neuron in self.output_neuron:
            neuron.activation = 0.0


    ## Adding list of columns to receptive field ##
    def add_to_receptive_field(self, column_list, field_type):
        if field_type == 'body':
            for column in column_list:
                if column not in self.body_receptive_field:
                    self.body_receptive_field = self.body_receptive_field + [column]

        if field_type == 'dendrite':
            for column in column_list:
                if column not in self.dendrite_receptive_field:
                    self.dendrite_receptive_field = self.dendrite_receptive_field + [column]

class Inhibition_Neuron:
    def __init__(self, column = None):
        self.activation = 0.0
        self.pool = []
        self.input_connections = []
        self.column = column
        self.comment = ''
        self.stored_activation = 0.0
        self.stored_max = 0.0
        self.stored_count = 0
class Interneuron():
    def __init__(self, dendrites_number, column = None):
        self.trace_activation = 0.0
        self.activation = 0.0
        self.max = 0.0
        self.stored_activation = 0.0
        self.stored_max = 0.0
        self.stored_count = 0
        self.number_of_activations = 0
        self.pool = np.array([])
        self.dendrites = []
        self.column = column
        self.prime = 0.0
        self.activation_function = 100.0
        self.threshold = default.activation_function_threshold
        self.neuron_threshold = default.neuron_threshold
        self.comment = ''

        for i in range(dendrites_number):
            dendrite = Dendrite()
            dendrite.neuron = self
            self.dendrites = self.dendrites + [dendrite]

        self.input_connections = []
        self.output_connections = []

    ## Adding OUTPUT of another column to interneuron pool
    def add_output_to_pool(self, backward_output_neuron):

        if self.pool.size == 0:
            self.pool = np.atleast_2d([[backward_output_neuron, default.to_pool_start]])
        else:
            self.pool = np.vstack(self.pool, np.atleast_2d([[backward_output_neuron, default.to_pool_start]]))
        print(backward_output_neuron.comment + ' added to ' +  self.comment + ' pool')
class Dendrite():
    def __init__(self, column = None):
        self.column = column
        self.neuron = None
        self.pool = np.atleast_2d([])
        self.input_connections = []
        self.activation = 0.0
        self.stored_activation = 0.0
        self.stored_count = 0
        self.threshold = default.dendrite_threshold
        self.threshold_percent = default.dendrite_threshold_percent

    def add_interneuron_to_pool(self, backward_interneuron):
        if self.pool.size == 0:
            self.pool = np.atleast_2d([[backward_interneuron, default.to_pool_start]])
        else:
            self.pool = np.vstack(self.pool, np.atleast_2d([[backward_interneuron, default.to_pool_start]]))
        print(backward_interneuron.comment + ' added to ' +  self.comment + ' pool')
class Output_Neuron():
    def __init__(self, column = None):
        self.activation = 0.0
        self.neuron_threshold = default.neuron_threshold
        self.stored_activation = 0.0
        self.stored_max = 0.0
        self.stored_count = 0
        self.activation_count = 0
        self.output_connections = []
        self.input_connections = []
        self.column = column
        self.comment = ''
class Sensor():
    def __init__(self, comment = '', cognit = None):
        self.comment = comment
        self.activation = 0.0
        self.output_connections = []
        self.cognit = cognit

    def feed_forward(self):
        for connection in self.output_connections:
            if self.activation != 0:
                connection.output.stored_activation += self.activation * connection.weight
                connection.output.stored_count += 1
                if connection.output.column not in self.cognit.activation_list:
                    self.cognit.activation_list = self.cognit.activation_list + [connection.output.column]

        # record_connections(self, record_inputs=False, record_outputs=True)

class Connection():
    def __init__(self, n_input, n_output, durability = default.durability):
        self.input = n_input
        self.output = n_output
        self.output_column = self.output.column
        self.weight = default.weight
        self.durability = durability
        self.input_record = 0.0
        self.output_record = 0.0
        self.activation_flag = False
# def record_connections(item, record_inputs, record_outputs):
#     if record_inputs == True:
#         if len(item.input_connections) != 0:
#             for connection in item.input_connections:
#                 connection.input_record = connection.input.activation
#                 connection.output_record = connection.output.activation
#     if record_outputs == True:
#         if len(item.output_connections) != 0:
#             for connection in item.output_connections:
#                 connection.input_record = connection.input.activation
#                 connection.output_record = connection.output.activation

