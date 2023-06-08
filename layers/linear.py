
import numpy as np
from typing import Tuple


class Layer:

    def has_parameters(self) -> bool:
        "Return True if a layer is trainable"
        return True
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplemented
    
    def backward(self, grad : np.ndarray) -> np.ndarray:
        "Calculate derivative"
        raise NotImplemented
        
    

class Linear(Layer):

    def __init__(self, input : int, output : int, init_fn : object) -> None:
        "Initialize weights"
        
        self.w = None
        self.b = None

        self.dw = 0
        self.db = 0

        # initialize parameters randomly
        self.w, self.b = init_fn(input, output)

    def has_parameters(self) -> bool:
        "Return True if a layer is trainable"
        return True
    
    def forward(self,x: np.ndarray) -> np.ndarray:
        "Multiply input with weights and bias"
        self.x = x
        return x@self.w + self.b

    def backward(self,grad: np.ndarray) -> np.ndarray:
        "Calculate derivative"

        self.dw = self.x.T@grad/grad.shape[0]
        self.db = grad.mean(axis=0)
        return grad@self.w.T
    
    def update_parameters(self,updates: Tuple[np.ndarray] ) -> None:
        "Update parameters, Recieves updates"

        self.w -= updates[0]
        self.b -= updates[1]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        "Enable object to be called as a method"
        return self.forward(x)

    def zero_grad(self) -> None:
        "Reset gradient"
        self.dw = 0
        self.db = 0

    def get_derivatives(self):
        "Return derivatives"
        return self.dw, self.db
    
    def get_parameteters(self):
        "Return parameters"
        return self.w, self.b