import pickle
import numpy as np
import pandas as pd
from classes import Cognit, Column, Connection, Sensor, connect
from utility import show_connections


class Sensor_Pattern:
    def __init__(self):
        self.pattern = []

    def add_to_pattern(self, add_list):
        if len(self.pattern) != 0:
            if len(add_list) == len(self.pattern[0]):
                self.pattern = self.pattern + [add_list]
        else:
            self.pattern = self.pattern + [add_list]

    def get_sensor_pattern(self, iteration_step):
        if iteration_step < len(self.pattern):
            return self.pattern[iteration_step]
        else:
            step = iteration_step % len(self.pattern)
            return self.pattern[step]

if __name__ == "__main__":
    sensor_pattern = Sensor_Pattern()
    sensor_pattern.add_to_pattern([0.9, 0.9, 0.4, 0.4, 0.0])
    # sensor_pattern.add_to_pattern([0.2, 0.2, 0.2, 0.2])

    with open('current_pattern.pickle', 'wb') as f:
        pickle.dump(sensor_pattern, f)
