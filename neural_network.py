import numpy as np

class Dense:
    def __init__(self, neruon_count, prev_neuron_count):
        self.neuron_count = neruon_count
        self.b = [np.zeros(neruon_count)]
        self.prev_neuron_count = prev_neuron_count