import numpy as np
import math

class LogisticRegression:
    def __init__(self, x, y, decision_boundary = 0.5, learning_rate = 0.01):
        self.w = np.zeros(len(x[0]))
        self.b = 0
        self.x = np.array(x)
        self.y = np.array(y)
        self.decision_boundary = decision_boundary
        self.learning_rate = learning_rate

    def z(self, x):
        return np.dot(self.w,x) + self.b
    
    def sigmoid(self, z):
        return 1/(1 - math.e**(-z))
    
    def f(self, x):
        return self.sigmoid(self.z(x))
    
    def cost(self):
        cost = 0
        for i in range(len(self.x)):
            fval = self.f(self.x[i])
            cost += -self.y[i]*math.log(fval) - (1 - self.y[i])*math.log(1 - fval)
        return cost
    
    def dcostdb(self):
        d = 0
        for i in range(len(self.x)):
            d += self.f(self.x[i]) - self.y[i]
        d /= len(self.x)
        return d
    
    def dcostdw(self):
        d_arr = np.zeros(len(self.w))
        for i in range(len(self.w)):
            d = 0
            for j in range(len(self.x)):
                d += (self.f(self.x[i]) - self.y[i])*self.x[j][i]
            d_arr[i] = d
        return d_arr

    
    def predict(self, x):
        if self.f(x) >= self.decision_boundary:
            return 1
        return 0