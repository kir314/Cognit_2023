import default

class Neuron():
    def __init__(self):

        self.activation = 0.0
        self.number_of_activations = 0
        self.dendrites = []
        self.inputs = []
        self.outputs = []
        self.pool = []
        self.threshold = default.activation_threshold
        self.activation_function = 0.0

class Dendrite():

    def __init__(self):
        self.pool = []
        self.inputs = []
        self.number_of_connections = 0
        self.activation = 0.0
        self.threshold = default.dendrite_threshold

class Connection():
    def __init__(self):
        self.input = None
        self.output = None
        self.weight = default.weight
        self.input_activation = 0.0
        self.output_activation = 0.0