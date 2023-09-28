import default
import numpy as np

class Cognit():
    def __init__(self):
        self.activation_list = []
        self.column = []
        self.layer = []
        self.sensor_layer = []
        self.new_connections = []
        self.joy = 0.0
        self.stress = 0.0

    def feed_sensors(self, signal_array_list):
        self.activation_list = []

        ## Feed sensors and add activated columns to column activation list ##

        for i in range(len(signal_array_list)):
            signal_array = signal_array_list[i]
            for sensor in self.sensor_layer[i].sensor:
                sensor.activation = signal_array[sensor.position[0]][sensor.position[1]]

                if len(sensor.output_connections) != 0:
                    for connection in sensor.output_connections:
                        if connection.output.column not in self.activation_list:
                            self.activation_list = self.activation_list + [
                                connection.output.column]
                        connection.output.stored_activation = connection.output.stored_activation + [
                            sensor.activation]
                        connection.strength += default.activation_strength_increase

    ## Feed forwarding sensors and columns in cognit ##
    def feed_forward(self):
        ## Feed forward columns in acctivation list ##
        if len(self.activation_list) != 0:
            i = 0
            while i < len(self.activation_list):
                self.activation_list[i].feed_forward()
                i += 1

            ## Walking back the list (optional) ##
            self.activation_list.reverse()
            for column in self.activation_list:
                column.feed_forward()

            for column in self.activation_list:
                column.learn()

            self.clean_up()

    ## Adding layer of equal columns to cognit ##
    def create_column_layer(self, comment, n_columns, n_interneuron, n_dendrites, y_size, n_outputs):
        layer = Layer(cognit = self, n_columns = n_columns, comment = comment, n_interneuron = n_interneuron, n_dendrites = n_dendrites, n_outputs = n_outputs, y_size=y_size)
        self.layer = self.layer + [layer]
        for column in layer.column:
            self.column = self.column + [column]

    def create_sensor_layer(self, comment, x_size, y_size = 1):
        sensor_layer = Sensor_Layer(cognit = self, comment = comment, x_size=x_size, y_size=y_size)
        self.sensor_layer = self.sensor_layer + [sensor_layer]

    ## Zeroing activations after feed forward ##
    def clean_up(self):
        # for column in self.activation_list:
        #     column.clean_up()
        self.new_connections = []

class Sensor_Layer():
    def __init__(self, cognit, comment, x_size, y_size = 1):
        self.sensor = []
        self.cognit = cognit
        self.comment = comment
        # self.check_1D_2D = check_1D_2D
        self.x_size = x_size
        self.y_size = y_size

        for i in range(y_size):
            for j in range(x_size):
                if y_size == 1: sensor_comment = comment + '_' + str(j)
                else: sensor_comment = comment + '_' + str(i) + '_' + str(j)
                sensor = Sensor(cognit=cognit, comment=sensor_comment)
                sensor.position = [i,j]
                self.sensor = self.sensor + [sensor]

class Layer():
    def __init__(self, cognit, comment, n_columns, n_interneuron, n_dendrites, n_outputs, y_size = 1):
        self.column = []
        self.cognit = cognit
        self.comment = comment
        self.x_size = int(n_columns / y_size)
        self.y_size = y_size
        for i in range(self.x_size):
            for j in range(y_size):
                column = Column(cognit, layer = self, interneuron_number = n_interneuron, dendrites_number = n_dendrites, outputs_number = n_outputs, comment = comment, position = [j,i])
                self.column = self.column + [column]

    def set_receptive_field_in_radius(self, radius):
        for column in self.column:
            column.add_to_receptive_field(method='in_radius', parameter = radius)

class Column():
    def __init__(self, cognit, layer, interneuron_number, dendrites_number, comment, outputs_number, position):

        self.position = position
        self.cognit = cognit
        self.layer = layer
        self.comment = comment + '_' + str(self.position)
        self.inhibition_neuron = Inhibition_Neuron(column = self)
        self.inhibition_neuron.comment = self.comment + '_block'
        self.output_neuron = []
        for i in range(outputs_number):
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

        self.column_receptive_field = []
        self.sensor_receptive_field = []

    ## Feed forward inside the column (winner interneuron takes all based on activation function) ##
    def feed_forward(self):

    ## Inhibition of interneuron: preinhibition + activation function calculation + inhibition ##

        ## Preinhibition ##
        if len(self.inhibition_neuron.stored_activation) != 0:

            #####################################
            ## Activation of inhibition neuron #
            if sum(self.inhibition_neuron.stored_activation) > self.inhibition_neuron.threshold:
                self.inhibition_neuron.activation = sum(self.inhibition_neuron.stored_activation)
            #####################################

            else: self.inhibition_neuron.activation = 0.0
            self.inhibition_neuron.stored_activation = []

        ## Calculating activation_function ##
        for neuron in self.interneuron:
            # Activating dendrites #
            neuron.prime = 0.0
            for dendrite in neuron.dendrites:
                if len(dendrite.input_connections) != 0:
                    if len(dendrite.stored_activation) != 0:

                        ####################################
                        ## Activation of dendrites #########
                        dendrite.activation = max(dendrite.stored_activation)
                        neuron.prime = max(neuron.prime, dendrite.activation)
                        dendrite.stored_activation = []
                        ####################################


            ############################################
            ## Activation of interneurons ##
            if len(neuron.stored_activation) != 0:
                if (sum(neuron.stored_activation) + neuron.prime - default.inhibitory_coefficient * neuron.column.inhibition_neuron.activation) > neuron.threshold:
                    neuron.activation = default.suppression * (max(neuron.stored_activation) - default.inhibitory_coefficient * neuron.column.inhibition_neuron.activation)
                else:
                    neuron.activation = 0.0
                neuron.stored_activation = []
            ############################################


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
            if neuron.activation != 0:
                if neuron.previous_activation >= default.memory_level:
                    neuron.activation = default.memory_level

                for connection in neuron.output_connections:
                    connection.output.stored_activation = connection.output.stored_activation + [neuron.activation + 0.01 * np.random.randn()]
                    connection.strength += default.activation_strength_increase
                    if connection.output.column not in self.cognit.activation_list:
                        self.cognit.activation_list = self.cognit.activation_list + [connection.output.column]

        for neuron in self.interneuron:
            neuron.previous_activation = neuron.activation

        ## Calculating activation of output neuron ##
        for neuron in self.output_neuron:
            if sum(neuron.stored_activation) > neuron.threshold:
                neuron.activation = max(neuron.stored_activation)
                for connection in neuron.output_connections:
                    connection.output.stored_activation = connection.output.stored_activation + [neuron.activation + 0.01 * np.random.randn()]
                    connection.strength += default.activation_strength_increase
                    if connection.output.column not in self.cognit.activation_list:
                        self.cognit.activation_list = self.cognit.activation_list + [connection.output.column]
                else: neuron.activation = 0.0
            neuron.stored_activation = []

    def learn(self):
        ## Learning connections output - interneuron ##
        for interneuron in self.interneuron:
            if interneuron.activation != 0:
                if len(self.column_receptive_field) != 0:
                    for column in self.column_receptive_field:
                        for output_neuron in column.output_neuron:
                            if output_neuron.activation != 0:

                                connection_flag = False
                                if len(interneuron.input_connections) != 0:
                                    for connection  in interneuron.input_connections:
                                        if connection.input == output_neuron: connection_flag = True

                                if connection_flag == True:
                                    neuron = Output_Neuron(column=column)
                                    neuron.comment = column.comment + '_out' + str(len(column.output_neuron))
                                    column.output_neuron = column.output_neuron + [neuron]
                                    connect(neuron, interneuron)
                                    self.cognit.new_connections = self.cognit.new_connections + [neuron.output_connections[-1]]
    def clean_up(self):
        # ## Cleaning up activations inside column after feed forward ##
        # self.inhibition_neuron.activation = 0.0
        # for neuron in self.interneuron:
        #     neuron.activation = 0.0
        # for neuron in self.output_neuron:
        #     neuron.activation = 0.0
        print(1)

    def add_to_receptive_field(self, method, parameter):
        if method == 'in_radius':
            for column in self.layer.column:
                if column != self:
                    position_delta = abs(column.position[0] - self.position[0]) + abs(column.position[1] - self.position[1])
                    if position_delta <= parameter:
                        self.column_receptive_field = self.column_receptive_field + [column]



class Inhibition_Neuron:
    def __init__(self, column = None):
        self.activation = 0.0
        self.threshold = default.neuron_threshold
        self.pool = []
        self.input_connections = []
        self.column = column
        self.comment = ''
        self.stored_activation = []

class Interneuron():
    def __init__(self, dendrites_number, column = None):
        self.trace_activation = 0.0
        self.activation = 0.0
        self.max = 0.0
        self.stored_activation = []
        self.number_of_activations = 0
        self.pool = np.array([])
        self.dendrites = []
        self.column = column
        self.prime = 0.0
        self.activation_function = 100.0
        self.threshold = default.activation_function_threshold
        self.neuron_threshold = default.neuron_threshold
        self.comment = ''
        self.previous_activation = 0.0

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
        self.comment = None
        self.column = column
        self.neuron = None
        self.pool = np.atleast_2d([])
        self.input_connections = []
        self.activation = 0.0
        self.stored_activation = []
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
        self.threshold = default.neuron_threshold
        self.stored_activation = []
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
        self.position = None

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
        self.strength = default.strength
        self.input_record = 0.0
        self.output_record = 0.0
        self.activation_flag = False


def connect(entity1, entity2):

    ready_flag = True
    for output_connection in entity1.output_connections:
        if output_connection.output  == entity2:
            ready_flag = False
    if ready_flag:
        connection = Connection(n_input=entity1, n_output=entity2, durability=1.0)
        entity2.input_connections = entity2.input_connections + [connection]
        entity1.output_connections = entity1.output_connections + [connection]
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

