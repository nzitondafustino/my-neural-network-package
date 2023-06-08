
from typing import Any


class BaseModel:

    def forward(self,x):
        raise NotImplemented
    
    def backward(self, grad):
        raise NotImplemented
    
    def __call__(self,x):
        return self.forward(x)
    
        
    def train(self):
        attributes = vars(self)
        for att in attributes:
            layer = attributes[att]
            if hasattr(layer, "_train"):
                layer.train()

    def eval(self):
        attributes = vars(self)
        for att in attributes:
            layer = attributes[att]
            if hasattr(layer, "_train"):
                layer.eval()
                