class Coordinate:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class SingleLinearRegression:

    def __init__(self, data, learning_rate = 0.01):
        self.data = []
        for i in data:
            self.data.append(Coordinate(i[0],i[1]))
        self.w = 0
        self.c = 0
        self.learning_rate = learning_rate

    def predict(self, x):
        return x*self.w + self.c
    
    def cost(self):
        cost = 0
        for p in self.data:
            cost += (self.predict(p.x) - p.y)**2
        cost /= 2*len(self.data)
        return cost
    
    def dcostdm(self):
        d = 0
        for p in self.data:
            d += -2*p.x*(p.y - self.predict(p.x))
        d /= len(self.data)
        return d
    
    def dcostdc(self):
        d = 0
        for p in self.data:
            d += -2*(p.y - self.predict(p.x))
        d /= len(self.data)
        return d
    
    def epoch(self):
        dcdm = self.dcostdm()
        dcostdc = self.dcostdc()
        self.w -= self.learning_rate*dcdm
        self.c -= self.learning_rate*dcostdc

    def iterate(self, repeat):
        for _ in range(repeat):
            self.epoch()
            print(f'cost {self.cost()}')
        print(f'y = {self.w}x + {self.c}')
        