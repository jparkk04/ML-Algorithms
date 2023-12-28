import numpy as np
import copy

class Normalize:
    def __init__(self, data):
        self.data = np.array(data, dtype=float)
        self.feature_array = self.data.T
        self.max_array = self.feature_array.max(axis=1)
        self.normalize_data()

class MaxNormalize(Normalize):
    def __init__(self, data):
        super().__init__(data)
    
    def normalize_data(self):
        new_array = copy.deepcopy(self.feature_array)
        new_array = (new_array.T / self.max_array).T
        self.normalized_array = new_array
        return new_array
    
    def get_normalized_array(self):
        return self.normalized_array
    
    def normalize_single_input(self,input):
        new_input = copy.deepcopy(input)
        for i in range(len(input)):
            new_input[i] /= self.max_array[i]