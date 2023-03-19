
from layers.linear import Linear
from activations.activations import ReLU
from utils.initialization import xavier_initialization

class MNISTModel:
    def __init__(self) -> None:
        self.linear1 = Linear(20,30,xavier_initialization)
        self.relu1 = ReLU()
        self.linear2 = Linear(30,5,xavier_initialization)

    def forward(self,x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        return x
    
    def backward(self,grad):
        grad = self.linear2.backward(grad)
        grad = self.relu1.backward(grad)
        grad = self.linear1.backward(grad)

    def __call__(self, x):
        return self.forward(x)