import numpy as np
from layers.linear import Layer

class Dropout(Layer):

    def __init__(self, probability : float) -> None:
        "initilize attributes"
        self.probability = probability

    def forward(self,x : np.ndarray, train : bool=True):
        "Run forward pass"
        if train: # use dropout if training
            self.dmat = np.random.binomial(1, 1 - self.probability,size=(x.shape))  # generate bernoulli random matrix
            return x * self.dmat/(1 - self.probability) # apply dropout and scale the output[to maintain expected value]
        else: # don't use dropout in test mode
            self.dmat = 1
            return x

    def backward(self, grad : np.ndarray):
        "Run backward pass"
        return grad * self.dmat

    def __call__(self, x, train=True):
        "Enable object to be called as a method"
        return self.forward(x, train)
    
    def has_parameters(self) -> bool:
        "Return True if a layer is trainable"
        return False