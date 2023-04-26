import default
import numpy as np
class Cognit():
    def __init__(self):
        self.column_activation_list = []
        self.sensor_activation_list = []
        self.sensor_list = []
        self.columns_list = []
        self.layer_list = []

    ## Creating sensor and adding it to list ##
    def create_sensor(self, comment = ''):
        sensor = Sensor(comment, self)
        self.sensor_list = self.sensor_list + [sensor]

    ## Getting signal list from signal constructor and feeding it to sensors ##
    def feed_sensors(self, signal_list):
        if len(signal_list) == len(self.sensor_list):
            for i in range(len(signal_list)):
                self.sensor_list[i].activation = signal_list[i]
        else: print('sensor inputs dont match')

    ## Sensors that have some signal go to sensor activation list ##
    def activate_sensors(self):
        for sensor in self.sensor_list:
            if sensor not in self.sensor_activation_list:
                self.sensor_activation_list = self.sensor_activation_list + [sensor]

    ## Adding layer of equal columns to cognit ##
    def add_layer(self, comment, n_columns, n_interneurons, n_dendrites):
        layer = Layer(cognit = self, n_columns = n_columns, comment = comment, n_interneurons = n_interneurons, n_dendrites = n_dendrites)
        self.layer_list = self.layer_list + [layer]
        for column in layer.columns_list:
            self.columns_list = self.columns_list + [column]

    ## Zeroing activations after feed forward ##
    def clean_up(self):
        for column in self.column_activation_list:
            column.clean_up()

    ## Feed forwarding sensors and columns in cognit ##
    def do_cycle(self):
        self.column_activation_list = []
        self.sensor_activation_list = []
        self.activate_sensors()
        for sensor in self.sensor_activation_list:
            sensor.feed_forward()
        if len(self.column_activation_list) != 0:
            i = 0
            while i < len(self.column_activation_list):
                self.column_activation_list[i].feed_forward()
                i += 1

    ## Learning weights in cognit ##
    def learn(self):
        for column in self.column_activation_list:
            index = self.column_activation_list.index(column)
            if index < (len(self.column_activation_list) - 1):
                for j in range(index, len(self.column_activation_list)):
                    column.add_to_pool(self.column_activation_list[j])

        for column in self.column_activation_list:
            column.adjust_real_durability()
            column.adjust_pool_durability()


class Layer():
    def __init__(self, cognit, comment, n_columns, n_interneurons, n_dendrites):
        self.columns_list = []
        self.cognit = cognit
        self.comment = comment
        for i in range(n_columns):
            column = Column(cognit, layer = self, interneurons_number = n_interneurons, dendrites_number = n_dendrites, comment = comment + '_' + str(i))
            self.columns_list = self.columns_list + [column]

class Column():
    def __init__(self, cognit, layer, interneurons_number, dendrites_number, comment):

        self.cognit = cognit
        self.layer = layer
        self.comment = comment
        self.inhibition_neuron = Inhibition_Neuron(column = self)
        self.inhibition_neuron.comment = self.comment + '_block'
        self.output_neuron = Output_Neuron(column = self)
        self.output_neuron.comment = self.comment + '_out'
        self.interneurons = []
        for i in range(interneurons_number):
            neuron = Interneuron(dendrites_number = dendrites_number, column = self)
            neuron.comment = self.comment + '_i' + str(i)
            for j in range(len(neuron.dendrites)):
                neuron.dendrites[j].comment = neuron.comment + '_d' + str(j)
                neuron.dendrites[j].column = self
            self.interneurons = self.interneurons + [neuron]

        self.body_receptive_field = []
        self.dendrite_receptive_field = []

    ## Feed forward inside the column (winner interneuron takes all based on activation function) ##
    def feed_forward(self):

        self.output_neuron.activation = 0.0
        self.output_neuron.activation_count = 0
        self.inhibit()
        self.activate_interneurons()
        self.activate_output()

    ## Inhibition of interneurons: preinhibition + activation function calculation + inhibition ##
    def inhibit(self):
        ## Preinhibition ##
        if self.inhibition_neuron.stored_count != 0:
            self.inhibition_neuron.activation = self.inhibition_neuron.stored_activation / self.inhibition_neuron.stored_count
            self.inhibition_neuron.stored_activation = 0.0
            self.inhibition_neuron.stored_count = 0

        if self.inhibition_neuron.activation != 0:
            for neuron in self.interneurons:
                neuron.stored_activation -= self.inhibition_neuron.activation * 0.5 ##### Add coefficient later
                neuron.stored_activation = max(neuron.stored_activation, 0.0)

        ## Calculating activation_function ##
        for neuron in self.interneurons:
            # Activating dendrites #
            neuron.prime = 0.0 ## can be changed for prime degradation
            for dendrite in neuron.dendrites:
                if len(dendrite.input_connections) != 0:
                    if dendrite.stored_count != 0:
                        dendrite.activation = dendrite.stored_activation
                        dendrite.stored_activation = 0.0
                        dendrite.stored_count = 0
                    ## setting threshold as medium activation required to fire ##
                    if ((dendrite.activation / len(dendrite.input_connections)) > dendrite.threshold_percent) and (dendrite.activation > dendrite.threshold):
                        neuron.prime = max(neuron.prime, dendrite.activation)
                    else:
                        dendrite.activation = 0.0
                    record_connections(dendrite, record_inputs=True, record_outputs=False)

            if neuron.stored_count != 0:
                neuron.activation = neuron.stored_activation / neuron.stored_count
                neuron.stored_activation = 0.0
                neuron.stored_count = 0
            # Calculating activation function for inhibition #
            if ((neuron.activation + neuron.prime) > neuron.threshold) & (neuron.activation != 0):
                neuron.activation_function = (neuron.threshold - neuron.prime) / neuron.activation
            else:
                neuron.activation_function = 100.0
                neuron.activation = 0.0

        ## Inhibition based on activation function ##
        A_min = 100.0
        neuron_min = None
        for neuron in self.interneurons:
            if neuron.activation_function < A_min:
                A_min = neuron.activation_function
                neuron_min = neuron
        if neuron_min != None:
            self.inhibition_neuron.activation = neuron_min.activation ## add coefficient later
            for neuron in self.interneurons:
                if neuron != neuron_min:
                    neuron.activation -= self.inhibition_neuron.activation
                    neuron.activation = max(0.0,neuron.activation)

    ## Calculating activation of interneurons after inhibition ##
    def activate_interneurons(self):
        for neuron in self.interneurons:
            if neuron.activation > neuron.neuron_threshold:
                for connection in neuron.output_connections:
                    connection.output.stored_activation += neuron.activation * connection.weight
                    connection.output.stored_count += 1
                    if connection.output.column not in self.cognit.column_activation_list:
                        self.cognit.column_activation_list = self.cognit.column_activation_list + [connection.output.column]
                self.output_neuron.activation += neuron.activation
                self.output_neuron.activation_count += 1
            else:
                neuron.activation = 0.0
            record_connections(neuron, record_inputs=True, record_outputs=True)

    ## Calculating activation of output neuron ##
    def activate_output(self):
        if self.output_neuron.activation != 0:
            for connection in self.output_neuron.output_connections:
                connection.output.stored_activation += self.output_neuron.activation ##+ np.random.randn()*0.05 (если будут волны)
                connection.output.stored_count += 1
                if connection.output.column not in self.cognit.column_activation_list:
                    self.cognit.column_activation_list = self.cognit.column_activation_list + [connection.output.column]
        record_connections(self.output_neuron, record_inputs=False, record_outputs=True)

    ## Cleaning up activations inside column after feed forward ##
    def clean_up(self):
        self.inhibition_neuron.activation = 0.0
        for neuron in self.interneurons:
            neuron.activation = 0.0
        self.output_neuron.activation = 0.0

    ## Add column from receptive field to pool of column (to interneurons) ##
    def add_to_pool(self, forward_column):
        if self in forward_column.body_receptive_field and self.output_neuron.activation != 0:
            for neuron in forward_column.interneurons:
                if neuron.pool.size != 0:
                    if (self.output_neuron not in neuron.pool[:,0]) and neuron.activation != 0:
                        not_connected_already = True
                        for connection in neuron.input_connections:
                            if connection.input == self.output_neuron: not_connected_already = False
                        if not_connected_already:
                            neuron.add_output_to_pool(self.output_neuron)
                else:
                    if neuron.activation != 0:
                        not_connected_already = True
                        for connection in neuron.input_connections:
                            if connection.input == self.output_neuron: not_connected_already = False
                        if not_connected_already:
                            neuron.add_output_to_pool(self.output_neuron)

        if self in forward_column.dendrite_receptive_field:
            for neuron in forward_column.interneurons:
                for dendrite in neuron.dendrites:
                    if dendrite.activation != 0 or neuron.activation != 0:
                        for backward_interneuron in self.interneurons:
                            if dendrite.pool.size != 0:
                                if (backward_interneuron not in dendrite.pool[:, 0]):
                                    not_connected_already = True
                                    for connection in dendrite.input_connections:
                                        if connection.input == backward_interneuron: not_connected_already = False
                                    if not_connected_already:
                                        dendrite.add_interneuron_to_pool(backward_interneuron)
                            else:
                                not_connected_already = True
                                for connection in dendrite.input_connections:
                                    if connection.input == backward_interneuron: not_connected_already = False
                                if not_connected_already:
                                    dendrite.add_interneuron_to_pool(backward_interneuron)

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


    def adjust_real_durability(self):
        for neuron in self.interneurons:
            if len(neuron.input_connections) != 0 and neuron.activation != 0:
                for connection in neuron.input_connections:
                    if connection.input.activation != 0:
                        connection.durability += default.up_durability
                    else:
                        connection.durability -= default.down_durability
                    if connection.durability < default.pool_to_connection_threshold:
                        neuron.add_output_to_pool(connection.input)
                        print(connection.input.comment + ' disconnected from ' + connection.output.comment)
                        connection_index = neuron.input_connections.index(connection)
                        connection.durability = 0.0
                        del neuron.input_connections[connection_index]

            for dendrite in neuron.dendrites:
                if len(dendrite.input_connections) != 0:
                    for connection in dendrite.input_connections:
                        activation_pattern = [neuron.activation !=0, dendrite.activation !=0, connection.input.activation != 0]
                        if (activation_pattern == [True, True, True]) or (activation_pattern == [True, False, True]) or (activation_pattern == [False, True, True]):
                            connection.durability += default.up_durability
                        elif (activation_pattern == [True, True, False]) or (activation_pattern == [False, True, False]) or (activation_pattern == [False, False, True]): \
                            connection.durability -= default.down_durability

                        if connection.durability < default.pool_to_connection_threshold:
                            dendrite.add_interneuron_to_pool(connection.input)
                            print(connection.input.comment + ' disconnected from ' + connection.output.comment)
                            connection_index = dendrite.input_connections.index(connection)
                            connection.durability = 0.0
                            del dendrite.input_connections[connection_index]

    def adjust_pool_durability(self):
        for neuron in self.interneurons:
            if len(neuron.pool) != 0 and neuron.activation != 0:
                for i in range(neuron.pool.shape[0]):
                    if neuron.pool[i,0].activation != 0:
                        neuron.pool[i,1] += default.up_durability
                    else:
                        neuron.pool[i,1] -= default.down_durability
                    if neuron.pool[i,1] > default.pool_to_connection_threshold:
                        connect(neuron.pool[i,0], neuron)
                        print(neuron.pool[i,0].comment + ' connected to ' + neuron.comment)
                        neuron.pool = np.delete(neuron.pool, i , axis = 0)
                    elif neuron.pool[i,1] < 0:
                        print(neuron.pool[i,0].comment + ' deleted from ' + neuron.comment + ' pool')
                        neuron.pool = np.delete(neuron.pool, i , axis = 0)

            for dendrite in neuron.dendrites:
                if dendrite.pool.size != 0:
                    for j in range(dendrite.pool.shape[0]):
                        activation_pattern = [neuron.activation != 0, dendrite.activation != 0, dendrite.pool[j, 0].activation != 0]
                        if (activation_pattern == [True, True, True]) or (activation_pattern == [True, False, True]) or (
                                activation_pattern == [False, True, True]):
                            dendrite.pool[j, 1] += default.up_durability
                        elif (activation_pattern == [True, True, False]) or (activation_pattern == [False, True, False]) or (activation_pattern == [False, False, True]): \
                            dendrite.pool[j, 1] -= default.down_durability

                        if dendrite.pool[j, 1] > default.pool_to_connection_threshold:
                            connect(dendrite.pool[j, 0], dendrite)
                            print(dendrite.pool[j, 0].comment + ' connected to ' + dendrite.comment)
                            dendrite.pool = np.delete(dendrite.pool, j, axis=0)
                        elif dendrite.pool[j, 1] < 0:
                            print(dendrite.pool[j, 0].comment + ' deleted from ' + dendrite.comment + ' pool')
                            dendrite.pool = np.delete(dendrite.pool, j, axis=0)



class Inhibition_Neuron:
    def __init__(self, column = None):
        self.activation = 0.0
        self.pool = []
        self.input_connections = []
        self.column = column
        self.comment = ''
        self.stored_activation = 0.0
        self.stored_count = 0
class Interneuron():
    def __init__(self, dendrites_number, column = None):
        self.activation = 0.0
        self.stored_activation = 0.0
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

    ## Adding OUTPUT of another column to INTERNEURONS pool
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
        self.activation_count = 0
        self.output_connections = []
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
                if connection.output.column not in self.cognit.column_activation_list:
                    self.cognit.column_activation_list = self.cognit.column_activation_list + [connection.output.column]

        record_connections(self,record_inputs=False, record_outputs=True)
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
def record_connections(item, record_inputs, record_outputs):
    if record_inputs == True:
        if len(item.input_connections) != 0:
            for connection in item.input_connections:
                connection.input_record = connection.input.activation
                connection.output_record = connection.output.activation
    if record_outputs == True:
        if len(item.output_connections) != 0:
            for connection in item.output_connections:
                connection.input_record = connection.input.activation
                connection.output_record = connection.output.activation
def connect(entity1, entity2, weight = default.weight):

    connection = Connection(n_input=entity1, n_output=entity2, durability=1.0)
    connection.weight = weight
    entity2.input_connections = entity2.input_connections + [connection]
    entity1.output_connections = entity1.output_connections + [connection]