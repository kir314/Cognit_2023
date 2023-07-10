import pickle
from classes import Cognit, Column, Connection, Sensor
from utility import connect

if __name__ == "__main__":
   cognit = Cognit()
   cognit.create_sensor_layer(comment = 'sensor_pos', check_1D_2D='1D', x_size = 3)
   cognit.create_sensor_layer(comment = 'motivation', check_1D_2D = '1D', x_size = 1)
   cognit.create_column_layer(comment = 'c_pos', n_columns = 3, n_interneuron = 2, n_dendrites = 3)
   cognit.create_column_layer(comment = 'c_turn', n_columns = 1, n_interneuron = 2, n_dendrites = 3)

   connect(cognit.sensor[0].sensor[0], cognit.column[0].interneuron[0])
   connect(cognit.sensor[0].sensor[1], cognit.column[1].interneuron[0])
   connect(cognit.sensor[0].sensor[2], cognit.column[2].interneuron[0])
   connect(cognit.sensor[1].sensor[0], cognit.column[2].interneuron[0])

   connect(cognit.column[0].interneuron[0], cognit.column[0].output_neuron[0])
   connect(cognit.column[2].interneuron[0], cognit.column[2].output_neuron[0])

   connect(cognit.column[0].output_neuron[0], cognit.column[1].interneuron[0])
   connect(cognit.column[3].output_neuron[0], cognit.column[1].interneuron[0])




   with open('current_cognit.pickle', 'wb') as f:
      pickle.dump(cognit, f)
