import numpy as np
import copy

class Normalize:
    def __init__(self, data):
        self.data = np.array(data)
        self.feature_array = np.array([])
        for i in range(len(data[0])):
            single_feature = np.array([])
            for j in range(len(data)):
                np.insert(single_feature, single_feature.size, data[j][i])
            np.insert(self.feature_array, self.feature_array.size ,single_feature)
        max_array = np.array([])
        for i in range(len(self.feature_array)):
            np.insert(max_array, max_array.size, max(self.feature_array[i]))
        self.normalized_array = self.normalize_data()

class MaxNormalize(Normalize):
    def __init__(self, data):
        super().__init__(data)
    
    def normalize_data(self):
        new_array = copy.deepcopy(self.feature_array)
        for i in range(len(new_array)):
            new_array[i] /= self.max_array[i]
        return new_array
    
    def get_normalized_array(self):
        return self.normalized_array
    
    def normalize_single_input(self,input):
        new_input = copy.deepcopy(input)
        for i in range(len(input)):
            new_input[i] /= self.max_array[i]
