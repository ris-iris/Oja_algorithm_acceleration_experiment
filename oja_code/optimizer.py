# Optimizers
class SGD:
    def __init__(self, lr=1e-2):
        self.lr = lr
    
    def step(self, w, A):
        return w + self.lr * A @ w
    
    def info(self):
        return f'SGD with constant lr={self.lr}'
    
class SGD_with_decreasing_step:
    def __init__(self, lr=1e-2, beta=1):
        self.lr = lr
        self.beta = beta
        self.t = 0
    
    def step(self, w, A):
        temp_lr = self.lr/(self.t + self.beta)
        self.t += 1
        
        return w + temp_lr * A @ w
    
    def info(self):
        return f'SGD with stepsize decrease lr/(beta + t) at every step, lr={self.lr}, beta={self.beta}'
    
class AcSGD:
    def __init__(self, alpha=1e-2, beta=1e-2):
        self.alpha = alpha
        self.beta = beta
        self.z = None
        self.t = 0
    
    def step(self, w, A):
        if self.t == 0:
            self.z = w
        
        y = w + self.beta * A @ w
        self.z = self.z + self.alpha * (self.t + 1) * A @ w
        
        return ((self.t + 1) * y + self.z) / (self.t + 2)
    
    def info(self):
        return f'AcSGD with alpha={self.alpha}, beta={self.beta}'   

class AcSGD_with_decreasing_step:
    def __init__(self, alpha=1e-2, beta=1e-2, gamma=1, decrease_mod='1/t'):
        def alpha_gen(alpha, gamma=gamma):
            t = 0
            while True:
                t += 1
                yield alpha/(t**gamma)
                
        def beta_gen(beta, gamma=gamma):
            t = 0
            while True:
                t += 1
                yield beta/(t**gamma)
                
        self.alpha_gen = alpha_gen(alpha)
        self.beta_gen = beta_gen(beta)
        self.alpha = alpha
        self.beta = beta
        self.z = None
        self.t = 0
    
    def step(self, w, A):
        if self.t == 0:
            self.z = w
        
        y = w + next(self.beta_gen) * A @ w
        self.z = self.z + next(self.alpha_gen) * (self.t + 1) * A @ w
        
        return ((self.t + 1) * y + self.z) / (self.t + 2)
    
    def info(self):
        return f'AcSGD with alpha={self.alpha}, beta={self.beta}'