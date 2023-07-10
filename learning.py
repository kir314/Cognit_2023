from classes import Cognit, Layer, Column, Connection, Dendrite, Inhibition_Neuron, Interneuron, Output_Neuron
from utility import connect
import numpy as np
import default

def learn(cognit):
    for column in cognit.activation_list:
        index = cognit.activation_list.index(column)
        if index < (len(cognit.activation_list) - 1):
            for j in range(index, len(cognit.activation_list)):
                add_to_pool(column, cognit.activation_list[j])

    for column in cognit.activation_list:
        adjust_real_durability(column)
        adjust_pool_durability(column)

 ## Add column from receptive field to pool of column (to interneuron) ##

def add_to_pool(column, forward_column):
    if column in forward_column.body_receptive_field and column.output_neuron.activation != 0:
        for neuron in forward_column.interneuron:
            if neuron.pool.size != 0:
                if (column.output_neuron not in neuron.pool[:,0]) and neuron.activation != 0:
                    not_connected_already = True
                    for connection in neuron.input_connections:
                        if connection.input == column.output_neuron: not_connected_already = False
                    if not_connected_already:
                        neuron.add_output_to_pool(column.output_neuron)
            else:
                if neuron.activation != 0:
                    not_connected_already = True
                    for connection in neuron.input_connections:
                        if connection.input == column.output_neuron: not_connected_already = False
                    if not_connected_already:
                        neuron.add_output_to_pool(column.output_neuron)

    if column in forward_column.dendrite_receptive_field:
        for neuron in forward_column.interneuron:
            for dendrite in neuron.dendrites:
                if dendrite.activation != 0 or neuron.activation != 0:
                    for backward_interneuron in column.interneuron:
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

def adjust_real_durability(column):
    for neuron in column.interneuron:
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

def adjust_pool_durability(column):
    for neuron in column.interneuron:
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
