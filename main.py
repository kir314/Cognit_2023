import numpy as np
import pandas as pd
import pickle
from classes import Cognit, connect
from utility import show_connections, create_activations_history, record_activations_history, create_durability_history, record_durability_history
from sensors import Sensor_Pattern

if __name__ == "__main__":

    with open('current_cognit.pickle', 'rb') as f:
        cognit = pickle.load(f)

    with open('current_pattern.pickle', 'rb') as f:
        sensor_pattern = pickle.load(f)

    for_record_durability = [cognit.columns_list[0].output_neuron.output_connections[0],
                             cognit.columns_list[1].output_neuron.output_connections[0],
                             cognit.columns_list[2].output_neuron.output_connections[0],
                             cognit.columns_list[3].output_neuron.output_connections[0],
                             cognit.columns_list[5].output_neuron.output_connections[0],
                             cognit.columns_list[5].output_neuron.output_connections[1],
                             cognit.columns_list[5].output_neuron.output_connections[2],
                             cognit.columns_list[5].output_neuron.output_connections[3],
                             cognit.columns_list[6].output_neuron.output_connections[0],
                             cognit.columns_list[6].output_neuron.output_connections[1],
                             cognit.columns_list[6].output_neuron.output_connections[2],
                             cognit.columns_list[6].output_neuron.output_connections[3],
                             ]
    durability_history = create_durability_history(for_record_durability)

    for_record_activations = [cognit.columns_list[0].output_neuron,
                              cognit.columns_list[1].output_neuron,
                              cognit.columns_list[2].output_neuron,
                              cognit.columns_list[3].output_neuron,
                              cognit.columns_list[4].output_neuron,
                              cognit.columns_list[5].output_neuron,
                              cognit.columns_list[6].output_neuron]
    activation_history = create_activations_history(for_record_activations)
    for i in range(20):
        signal_list = sensor_pattern.get_sensor_pattern(i)
        cognit.feed_sensors(signal_list)
        cognit.do_cycle()
        cognit.learn()
        durability_history = record_durability_history(durability_history, for_record_durability)
        activation_history = record_activations_history(activation_history, for_record_activations)
        cognit.clean_up()
    print(1)