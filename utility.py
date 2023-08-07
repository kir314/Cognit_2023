import numpy as np
import pandas as pd
from classes import Connection


def add_connection_array(connection, show_array, non_zero):
    if non_zero == True:
        if connection.input_record != 0 or connection.output_record != 0:
            if str(id(connection)) not in str(show_array[:,-1]):
                temp_array = np.atleast_2d([str(connection.input.comment),
                                      str(connection.output.comment),
                                      connection.input.activation,
                                      connection.output.activation,
                                      connection.output.stored_activation,
                                      id(connection)])
                show_array = np.append(show_array, temp_array, axis = 0)
            else:
                index = np.where(str(id(connection)) == show_array[:,-1])[0][0]
                show_array[index,2] = connection.input.activation
                show_array[index,3] = connection.output.activation
                show_array[index,4] = connection.output.stored_activation

    else:
        if str(id(connection)) not in show_array[:, -1]:
            temp_array = np.atleast_2d([str(connection.input.comment),
                                        str(connection.output.comment),
                                        connection.input.activation,
                                        connection.output.activation,
                                        connection.output.stored_activation,
                                        id(connection)])
            show_array = np.append(show_array, temp_array, axis=0)
        else:
            index = np.where(str(id(connection)) == show_array[:,-1])[0][0]
            show_array[index,2] = connection.input.activation
            show_array[index,3] = connection.output.activation
            show_array[index,4] = connection.output.stored_activation

    return(show_array)

def show_connections(cognit, non_zero = True):
    show_array = np.atleast_2d(['input', 'output','in_activ','out_activ','stored','id'])
    for sensor in cognit.sensor_activation_list:
        for connection in sensor.output_connections:
            show_array = add_connection_array(connection, show_array, non_zero)
    for column in cognit.column_activation_list:
        for connection in column.inhibition_neuron.input_connections:
            show_array = add_connection_array(connection, show_array, non_zero)
        for interneuron in column.interneuron:
            for connection in interneuron.input_connections:
                show_array = add_connection_array(connection, show_array, non_zero)
            for connection in interneuron.output_connections:
                show_array = add_connection_array(connection, show_array, non_zero)
        for connection in column.output_neuron.output_connections:
            show_array = add_connection_array(connection, show_array, non_zero)
    return(show_array)

def create_activations_history(items_list):
    columns = []
    for item in items_list:
        columns = columns + [item.comment]
    history = pd.DataFrame(columns = columns)
    return history

def record_activations_history(history, items_list):
    columns = []
    activations = []
    for item in items_list:
        columns = columns + [item.comment]
        activations = activations + [item.activation]
    temp = pd.DataFrame(columns = columns, data = [activations])
    history = pd.concat([history, temp])
    return history

def create_durability_history(connections_list):
    columns = []
    for item in connections_list:
        columns = columns + [item.input.comment + ' : ' + item.output.comment]
    history = pd.DataFrame(columns = columns)
    return history

def record_durability_history(history, connections_list):
    columns = []
    durabilities = []
    for item in connections_list:
        columns = columns + [item.input.comment + ' : ' + item.output.comment]
        durabilities = durabilities + [item.durability]
    temp = pd.DataFrame(columns = columns, data = [durabilities])
    history = pd.concat([history, temp])
    return history

def connect(entity1, entity2):

    ready_flag = True
    for output_connection in entity1.output_connections:
        if output_connection.output  == entity2:
            ready_flag = False
    if ready_flag:
        connection = Connection(n_input=entity1, n_output=entity2, durability=1.0)
        entity2.input_connections = entity2.input_connections + [connection]
        entity1.output_connections = entity1.output_connections + [connection]

def disconnect(entity1, entity2):
    if len(entity1.output_connections) != 0:
        for connection in entity1.output_connections:
            if connection.output == entity2:
                out_index = entity1.output_connections.index(connection)
                del entity1.output_connections[out_index]
        for connection in entity2.input_connections:
            if connection.input == entity1:
                in_index = entity2.input_connections.index(connection)
                del entity2.input_connections[in_index]


def connect_inter_to_output(layer):
    for column in layer.column:
        for interneuron in column.interneuron:
            for output_neuron in column.output_neuron:
                connect(interneuron, output_neuron)

def inhibit_in_radius(layer, distance_threshold):
    for column in layer.column:
        for another_column in layer.column:
            if column != another_column:
                distance = (column.position[0]-another_column.position[0]) ** 2 + (column.position[1]-another_column.position[1]) ** 2
                if distance > distance_threshold:
                    connect(column.interneuron[0], another_column.inhibition_neuron)
                    connect(another_column.interneuron[0], column.inhibition_neuron)

def connect_by_position(layer, position1, position2, type):
    column1 = None
    column2 = None
    for column in layer.column:
        if column.position == position1:
            column1 = column
        elif column.position == position2:
            column2 = column

        if column1 != None and column2 != None:
            if type == 'inh':
                connect(column1.interneuron[0], column2.inhibition_neuron)
            elif type == 'ord':
                connect(column1.output_neuron[0], column2.interneuron[0])

def connect_sensors_to_columns(sensor_layer, column_layer, type):
    if type == 'individual':
        for sensor in sensor_layer.sensor:
            for column in column_layer.column:
                if sensor.position == column.position:
                    connect(sensor, column.interneuron[0])