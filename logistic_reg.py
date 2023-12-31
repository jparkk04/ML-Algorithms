import numpy as np

class LogisticRegression:
    def __init__(self, x, y, learning_rate = 0.01):
        self.x = np.array(x)
        self.y = np.array(y)
        self.learning_rate = learning_rate