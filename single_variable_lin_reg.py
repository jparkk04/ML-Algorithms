class Coordinate:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class SingleLinearRegression:
    data = []

    def __init__(self, data, learning_rate = 0.01):
        for i in data:
            self.data.append(Coordinate(i[0],i[1]))
        self.m = 0
        self.c = 0
        self.learning_rate = learning_rate
    
    def cost(self):
        cost = 0
        for p in self.data:
            cost += (p.y - p.x*self.m + self.c)**2
        cost /= len(self.data)
        return cost
    
    def dcostdm(self):
        d = 0
        for p in self.data:
            d += -2*p.x*(p.y - p.x*self.m + self.c)
        d /= len(self.data)
        return d
    
    def dcostdc(self):
        d = 0
        for p in self.data:
            d += 2*(p.y - p.x*self.m + self.c)
        d /= len(self.data)
        return d
    
    def epoch(self):
        dcdm = self.dcostdm()
        dcostdc = self.dcostdc()
        self.m -= self.learning_rate*dcdm
        self.c -= self.learning_rate*dcostdc

    def iterate(self, repeat):
        for _ in range(repeat):
            self.epoch()
            print(f'cost {self.cost()}')
        print(f'y = {self.m}x + {self.c}')