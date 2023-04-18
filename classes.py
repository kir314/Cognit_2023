import numpy as np
import default
from utility import show_connections
class Cognit():
    def __init__(self):
        self.column_activation_list = []
        self.sensor_activation_list = []
        self.sensor_list = []
        self.columns_list = []
        self.layer_list = []
    def create_sensor(self, comment = ''):
        sensor = Sensor(comment, self)
        self.sensor_list = self.sensor_list + [sensor]

    def feed_sensors(self, signal_list):
        if len(signal_list) == len(self.sensor_list):
            for i in range(len(signal_list)):
                self.sensor_list[i].activation = signal_list[i]
        else: print('sensor inputs dont match')
    def activate_sensors(self):
        for sensor in self.sensor_list:
            if sensor not in self.sensor_activation_list:
                self.sensor_activation_list = self.sensor_activation_list + [sensor]

    def add_layer(self, comment, n_columns, n_interneurons, n_dendrites):
        layer = Layer(cognit = self, n_columns = n_columns, comment = comment, n_interneurons = n_interneurons, n_dendrites = n_dendrites)
        self.layer_list = self.layer_list + [layer]
        for column in layer.columns_list:
            self.columns_list = self.columns_list + [column]

    def clean_up(self):
        for column in self.column_activation_list:
            column.clean_up()
    def do_cycle(self):
        #activation
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

    def learn(self):
        for column in self.column_activation_list:
            index = self.column_activation_list.index(column)
            if index < (len(self.column_activation_list) - 1):
                for j in range(index, len(self.column_activation_list)):
                    column.add_to_pool(self.column_activation_list[j])


        # for column in self.column_activation_list:
        #     column.learn()
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
    def feed_forward(self):

        self.output_neuron.activation = 0.0
        self.output_neuron.activation_count = 0
        self.inhibit()
        self.activate_interneurons()
        self.activate_output()

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
    def activate_output(self):
        if self.output_neuron.activation != 0:
            for connection in self.output_neuron.output_connections:
                connection.output.stored_activation += self.output_neuron.activation
                connection.output.stored_count += 1
                if connection.output.column not in self.cognit.column_activation_list:
                    self.cognit.column_activation_list = self.cognit.column_activation_list + [connection.output.column]
        record_connections(self.output_neuron, record_inputs=False, record_outputs=True)

    def clean_up(self):
        self.inhibition_neuron.activation = 0.0
        for neuron in self.interneurons:
            neuron.activation = 0.0
        self.output_neuron.activation = 0.0

    def learn(self):
        for neuron in self.interneurons:
            if len(neuron.input_connections) != 0:
                for connection in neuron.input_connections:
                    if connection.output.activation != 0 and connection.input.activation != 0:
                        connection.durability += 0.01
                    elif connection.output.activation !=0 or connection.input.activation != 0:
                        connection.durability -= 0.01

    def add_to_pool(self, forward_column):
        if self in forward_column.body_receptive_field and self.output_neuron.activation != 0:
            for neuron in forward_column.interneurons:
                if len(neuron.pool) != 0:
                    if (self.output_neuron not in neuron.pool[:,0]) and neuron.activation != 0:
                        for connection in neuron.input_connections:
                            if connection.input != self.output_neuron:
                                neuron.add_output_to_pool(self.output_neuron)

    def add_to_receptive_field(self, column_list, field_type):
        if field_type == 'body':
            for column in column_list:
                if column not in self.body_receptive_field:
                    self.body_receptive_field = self.body_receptive_field + [column]
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
        self.pool = []
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

    def add_output_to_pool(self, backward_output_neuron):
        self.pool = self.pool + [[backward_output_neuron, default.to_pool_start]]
        print(self.comment + ' added to ' + backward_output_neuron.comment + ' pool')
class Dendrite():
    def __init__(self, column = None):
        self.column = column
        self.neuron = None
        self.pool = []
        self.input_connections = []
        self.activation = 0.0
        self.stored_activation = 0.0
        self.stored_count = 0
        self.threshold = default.dendrite_threshold
        self.threshold_percent = default.dendrite_threshold_percent
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