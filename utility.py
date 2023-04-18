import numpy as np
import pandas as pd


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
        for interneuron in column.interneurons:
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