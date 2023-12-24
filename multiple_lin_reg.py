import numpy as np

class MultipleLinearRegression:

    def __init__(self, x, y, learning_rate = 0.01):
        self.w = np.zeros(len(x[0]))
        self.b = 0
        self.y = np.array(y)
        self.x = np.array(x)
        self.learning_rate = learning_rate

    def calculate(self, xi):
        return np.dot(self.w,xi) + self.b
    
    def cost(self):
        cost = 0
        for i in range(len(self.x)):
            cost += (self.y[i] - self.calculate(self.x[i]))**2
        cost /= len(self.x)
        return cost