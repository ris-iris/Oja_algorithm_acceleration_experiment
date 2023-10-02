import matplotlib.pyplot as plt
import numpy as np
from oja_code.oja_utils import max_eigenvec

# class for running Oja's algorithm
class Oja:
    def __init__(self, d, optimizer, weight=None):
        if weight is None:
            self.weight = np.random.uniform(size=(d, 1))
            self.weight /= np.linalg.norm(self.weight, 2)
        else:
            self.weight = weight
        self.optimizer = optimizer
        self.weight_log = [self.weight, ]
        
    def update(self, A):
        self.weight = self.optimizer.step(self.weight, A)
        self.weight /= np.linalg.norm(self.weight, 2)
        self.weight_log.append(self.weight)
        
    def run(self, a_stream, max_iter = 1000):
        assert len(self.weight_log) == 1, 'Trying to run second time'
        for i in range(max_iter):
            A = next(a_stream)
            self.update(A)
        return self.weight
    
    def loss_history(self, reference_w):
        loss_h = []
        for w in self.weight_log:
            loss_h.append((1 - (w.T @ reference_w) ** 2).item())
        return loss_h
    
    def rayleigh_quotient_history(self, A0):
        rq_h = []
        for w in self.weight_log:
            rq_h.append((w.T @ A0 @ w).item())
        return rq_h
    
    def pretty_plot(self, A0, plot_mode=plt.loglog):
        real_w = max_eigenvec(A0)
        
        loss = self.loss_history(real_w)
        rq = self.rayleigh_quotient_history(A0)
        
        plt.figure(figsize=(14, 5))
        
        plt.subplot(1, 2, 1)
        plt.title('Rayleigh quotient')
        plot_mode(rq)
        plt.grid()
        
        plt.subplot(1, 2, 2)
        plt.title('sin^2(w, real_w)')
        plot_mode(loss)
        plt.grid()
        