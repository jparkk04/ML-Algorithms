import numpy as np
import copy

class Normalize:
    def __init__(self, data):
        self.data = np.array(data, dtype=float)
        self.feature_array = self.data.T

class MaxNormalize(Normalize):
    def __init__(self, data):
        super().__init__(data)
        self.max_array = self.feature_array.max(axis=1)
        self.normalize_data()
    
    def normalize_data(self):
        new_array = copy.deepcopy(self.feature_array)
        new_array = (new_array.T / self.max_array).T
        self.normalized_array = new_array
        return new_array
    
    def get_normalized_array(self):
        return self.normalized_array
    
    def normalize_single_input(self,input):
        normalized_input = input/self.max_array
        return normalized_input

class MeanNormalize(Normalize):
    def __init__(self, data):
        super().__init__(data)
        self.mean_array = self.feature_array.mean(axis=1)
        self.max_array = self.feature_array.max(axis=1)
        self.min_array = self.feature_array.min(axis=1)
        self.normalize_data()
    
    def normalize_data(self):
        new_array = copy.deepcopy(self.feature_array)
        new_array = ((new_array.T - self.mean_array)/(self.max_array - self.min_array)).T
        self.normalized_array = new_array
        return new_array
    
    def get_normalized_data(self):
        return self.normalized_array
    
    def normalize_single_input(self,input):
        normalized_input = (input - self.mean_array)/(self.max_array - self.min_array)
        return normalized_input
    
class ZScoreNormalize(Normalize):
    def __init__(self, data):
        super().__init__(data)
        self.mean_array = self.feature_array.mean(axis=1)
        self.std_array = self.feature_array.std(axis=1)
    def normalize_data(self):
        new_array = copy.deepcopy(self.feature_array)
        new_array = ((new_array.T - self.mean_array)/self.std_array).T
        self.normalized_array = new_array

    def get_normalized_data(self):
        return self.normalized_array
    
    def normalize_single_input(self,input):
        normalized_input = (input - self.mean_array)/self.std_array
        return normalized_input