
import numpy as np

class Linear:

    def __init__(self, input, output, init_fn) -> None:
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
    
    def update_parameters(self, learning_rate: float , multiplier: float =1.0) -> None:
        "Update parameters, If SGD is used the multiplier is 1. Multplier is different for other optimizers"
        self.w -= learning_rate * self.dw * multiplier
        self.b -= learning_rate * self.db * multiplier

    def __call__(self, x: np.ndarray) -> np.ndarray:
        "Enable object to be called as a method"
        return self.forward(x)

    def zero_grad(self) -> None:
        "Reset gradient"
        self.dw = 0
        self.db = 0
