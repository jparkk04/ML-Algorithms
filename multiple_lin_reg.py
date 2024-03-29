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
        cost = 0.0
        for i in range(len(self.x)):
            cost += (self.calculate(self.x[i]) - self.y[i])**2
        cost /= 2*len(self.x)
        return cost
    
    def dcostdb(self):
        d = 0.0
        for i in range(len(self.x)): #each datapoint
            d += (self.calculate(self.x[i]) - self.y[i])
        d /= len(self.x)
        return d
    
    def dcostdw(self):
        dlist = np.zeros(len(self.w))
        for i in range(len(self.w)):
            d = 0
            for j in range(len(self.x)):
                d += 2*(self.calculate(self.x[i]) - self.y[j])*self.x[j][i]
            d /= len(self.x)
            dlist[i] = d
        return dlist

    
    def epoch(self):
        dcdb = self.dcostdb()
        dcdw = self.dcostdw()
        self.w -= self.learning_rate*dcdw
        self.b -= self.learning_rate*dcdb

    def iterate(self, repeat):
        for _ in range(repeat):
            self.epoch()
            print(f'cost {self.cost()}')