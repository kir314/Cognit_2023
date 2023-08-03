import pickle
from classes import Cognit, Column, Connection, Sensor
from utility import connect, connect_inter_to_output, inhibit_in_radius, connect_by_position, connect_sensors_to_columns

if __name__ == "__main__":
   cognit = Cognit()
   cognit.create_sensor_layer(comment = 'sensor_pos', check_1D_2D='2D', x_size = 5, y_size = 5)
   cognit.create_sensor_layer(comment = 'motivation', check_1D_2D = '1D', x_size = 1)
   cognit.create_column_layer(comment = 'c_pos', n_columns = 25, n_interneuron = 1, n_dendrites = 1, n_outputs = 1, y_size = 5)
   cognit.create_column_layer(comment = 'move', n_columns = 1, n_interneuron = 4, n_dendrites = 1, n_outputs = 1, y_size = 1)


   connect(cognit.sensor_layer[0].sensor[0][1], cognit.column[5].interneuron[0])
   # connect(cognit.sensor_layer[1].sensor[0], cognit.column[4].interneuron[0])

   connect_inter_to_output(cognit.layer[0])
   connect_sensors_to_columns(cognit.sensor_layer[0], cognit.layer[0], 'individual')
   inhibit_in_radius(cognit.layer[0], 1)
   connect_by_position(cognit.layer[0], [2, 2], [2, 3], 'ord')
   connect_by_position(cognit.layer[0], [2, 3], [2, 4], 'ord')
   connect_by_position(cognit.layer[0], [2, 2], [2, 1], 'ord')
   connect_by_position(cognit.layer[0], [2, 1], [2, 0], 'ord')

   connect(cognit.sensor_layer[1].sensor[0], cognit.column[22].interneuron[0])


   with open('current_cognit.pickle', 'wb') as f:
      pickle.dump(cognit, f)
