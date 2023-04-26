import numpy as np
import pandas as pd
import pickle
from classes import Cognit
from utility import show_connections, create_activations_history, record_activations_history, create_durability_history, record_durability_history
from sensors import Sensor_Pattern
from learning import learn

if __name__ == "__main__":

    with open('current_cognit.pickle', 'rb') as f:
        cognit = pickle.load(f)

    with open('current_pattern.pickle', 'rb') as f:
        sensor_pattern = pickle.load(f)

    for_record_durability = []
    for_record_activations = []
    for column in cognit.columns_list:
        if len(column.output_neuron.output_connections) != 0:
            for j in range(len(column.output_neuron.output_connections)):
                for_record_durability = for_record_durability + [column.output_neuron.output_connections[j]]
                for_record_activations = for_record_activations + [column.output_neuron]

    durability_history = create_durability_history(for_record_durability)
    activation_history = create_activations_history(for_record_activations)
    for i in range(100):
        signal_list = sensor_pattern.get_sensor_pattern(i)
        cognit.feed_sensors(signal_list)
        cognit.do_cycle()
        learn(cognit)
        durability_history = record_durability_history(durability_history, for_record_durability)
        activation_history = record_activations_history(activation_history, for_record_activations)
        cognit.clean_up()
    print(1)