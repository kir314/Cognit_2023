import pickle
from classes import Cognit, Column, Connection, Sensor
from utility import connect, connect_inter_to_output, inhibit_in_radius, at_pos, connect_sensors_to_columns

if __name__ == "__main__":
   cognit = Cognit()
   cognit.create_sensor_layer(comment = 'sensor_pos', x_size = 5, y_size = 5)
   cognit.create_sensor_layer(comment = 'stimulus', x_size = 1)
   cognit.create_sensor_layer(comment = 'impatience', x_size = 1)
   cognit.create_column_layer(comment = 'c_pos', n_columns = 25, n_interneuron = 1, n_dendrites = 1, n_outputs = 1, y_size = 5)
   cognit.create_column_layer(comment = 'move', n_columns = 1, n_interneuron = 4, n_dendrites = 1, n_outputs = 1, y_size = 1)

   cognit.layer[0].set_receptive_field_in_radius(1)

   connect_inter_to_output(cognit.layer[0])
   connect_sensors_to_columns(cognit.sensor_layer[0], cognit.layer[0], 'individual')
   inhibit_in_radius(cognit.layer[0], 1)
   connect(at_pos(cognit.layer[0], [2,2]).output_neuron[0], at_pos(cognit.layer[0], [2,3]).interneuron[0])
   connect(at_pos(cognit.layer[0], [2, 2]).output_neuron[0], cognit.layer[1].column[0].interneuron[3])

   connect(at_pos(cognit.layer[0], [2, 2]).output_neuron[0], at_pos(cognit.layer[0], [2, 1]).interneuron[0])
   connect(at_pos(cognit.layer[0], [2, 2]).output_neuron[0], cognit.layer[1].column[0].interneuron[2])


   connect(at_pos(cognit.layer[0], [2,3]).output_neuron[0], at_pos(cognit.layer[0], [3,3]).interneuron[0])
   connect(at_pos(cognit.layer[0], [2, 3]).output_neuron[0], cognit.layer[1].column[0].interneuron[0])

   connect(at_pos(cognit.layer[0], [2, 1]).output_neuron[0], at_pos(cognit.layer[0], [1, 1]).interneuron[0])
   connect(at_pos(cognit.layer[0], [2, 1]).output_neuron[0], cognit.layer[1].column[0].interneuron[1])

   connect(cognit.sensor_layer[1].sensor[0], at_pos(cognit.layer[0], [0,0]).interneuron[0])

   cognit.layer[1].column[0].interneuron[0].comment = 'up'
   cognit.layer[1].column[0].interneuron[1].comment = 'down'
   cognit.layer[1].column[0].interneuron[2].comment = 'left'
   cognit.layer[1].column[0].interneuron[3].comment = 'right'

   connect(cognit.sensor_layer[1].sensor[0], at_pos(cognit.layer[0], [1,1]).interneuron[0])

   connect(cognit.sensor_layer[2].sensor[0], cognit.layer[1].column[0].interneuron[0])
   connect(cognit.sensor_layer[2].sensor[0], cognit.layer[1].column[0].interneuron[1])
   connect(cognit.sensor_layer[2].sensor[0], cognit.layer[1].column[0].interneuron[2])
   connect(cognit.sensor_layer[2].sensor[0], cognit.layer[1].column[0].interneuron[3])

   for interneuron in cognit.layer[1].column[0].interneuron:
      interneuron.neuron_threshold = 1.0
      interneuron.threshold = 1.0



   with open('current_cognit.pickle', 'wb') as f:
      pickle.dump(cognit, f)
