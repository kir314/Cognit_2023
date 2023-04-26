import pickle
from classes import Cognit, Column, Connection, Sensor, connect

if __name__ == "__main__":
   cognit = Cognit()
   cognit.create_sensor(comment='s0')
   cognit.create_sensor(comment='s1')
   cognit.create_sensor(comment='s2')
   cognit.create_sensor(comment='s3')
   cognit.create_sensor(comment='s4')
   cognit.add_layer(comment = 'c1', n_columns = 5, n_interneurons = 1, n_dendrites = 1)
   cognit.add_layer(comment = 'c2', n_columns = 5, n_interneurons = 1, n_dendrites = 1)

   s0 = cognit.sensor_list[0]
   s1 = cognit.sensor_list[1]
   s2 = cognit.sensor_list[2]
   s3 = cognit.sensor_list[3]
   s4 = cognit.sensor_list[4]

   c0 = cognit.columns_list[0]
   c1 = cognit.columns_list[1]
   c2 = cognit.columns_list[2]
   c3 = cognit.columns_list[3]
   c4 = cognit.columns_list[4]

   cp0 = cognit.columns_list[5]
   cp1 = cognit.columns_list[6]

   connect(s0, c0.interneurons[0])
   # connect(s1, c1.interneurons[0])
   # connect(s2, c2.interneurons[0])
   # connect(s3, c3.interneurons[0])
   # connect(s4, c4.interneurons[0])

   connect(c0.output_neuron, cp0.interneurons[0])
   connect(cp0.output_neuron, c1.interneurons[0])
   #
   # connect(c1.output_neuron, cp0.interneurons[0])
   #
   # connect(c2.output_neuron, cp1.interneurons[0])
   # connect(c3.output_neuron, cp1.interneurons[0])
   #
   # connect(cp0.output_neuron, c0.interneurons[0])
   # connect(cp0.output_neuron, c1.interneurons[0])
   # connect(cp0.output_neuron, c2.inhibition_neuron)
   # connect(cp0.output_neuron, c3.inhibition_neuron)
   #
   # connect(cp1.output_neuron, c2.interneurons[0])
   # connect(cp1.output_neuron, c3.interneurons[0])
   # connect(cp1.output_neuron, c0.inhibition_neuron)
   # connect(cp1.output_neuron, c1.inhibition_neuron)
   # connect(c4.output_neuron, cp0.interneurons[0])

   # cp0.add_to_receptive_field([c0,c1,c4], field_type='body')
   # cp1.add_to_receptive_field([c2,c3, c4], field_type='body')
   c1.add_to_receptive_field([c0], field_type = 'dendrite')

   with open('current_cognit.pickle', 'wb') as f:
      pickle.dump(cognit, f)
