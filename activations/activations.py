import numpy as np

class Activation:

    "Activation base class"

    def __call__(self,x : np.ndarray) -> np.ndarray:
        "Enable object to be called as a method"
        return self.forward(x)
        
    def forward(self,x : np.ndarray) -> np.ndarray:
        "Raise Excepetion when not implemented in sub-class"
        raise NotImplemented
    
    def backward(self,grad : np.ndarray=1) -> None:
        "Raise Excepetion when not implemented in sub-class"
        raise NotImplemented
    
    def has_parameters(self) -> bool:
        "Return True if a layer is trainable"
        return False
    

class ReLU(Activation):

    def __init__(self) -> None:
        "Call constructor of Parent class"
        super().__init__()

    def forward(self,x):
        self.x = x
        return np.clip(x, 0, None)
    
    def backward(self,grad=1):
        return grad * np.where(self.x > 0,1,0)
    

class Sigmoid(Activation):

    def __init__(self) -> None:
        "Call constructor of Parent class"
        super().__init__()

    def forward(self, x):
        "Caluculate Sigmoid"
        self.x = x
        return np.where(x < 0, np.exp(x)/(np.exp(x) + 1), 1/(1 + np.exp(-x)))
    
    def backward(self,grad=1):
        "Calculate derivative of Sigmoin"
        return grad * self.forward(self.x) * (1 - self.forward(self.x))
    
class Tanh(Activation):

    def __init__(self) -> None:
        "Call constructor of Parent class"
        super().__init__()

    def forward(self, x):
        "Caluculate Tanh"
        self.x = x
        return np.where(x < 0, 2 * np.exp(2*x)/(1 + np.exp(2*x)) - 1, 2/(1 + np.exp(-2*x)) - 1)
    
    def backward(self,grad=1):
        "Calculate delivative of Tanh"
        return grad * (1 - self.forward(self.x)**2)


# x = np.random.randn(32,10) * 1001
# act = Sigmoid()
# print(act(x))
# print(act.backward())