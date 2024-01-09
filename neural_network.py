import numpy as np

class Dense:
    def __init__(self, neruon_count):
        self.b = [np.zeros(neruon_count)]