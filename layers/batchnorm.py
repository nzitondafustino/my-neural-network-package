import numpy as np
from typing import Tuple
from layers.linear import Layer


class BacthNorm1d(Layer):

    def __init__(self,num_features : int, eps : float=1e-05, momentum: float=0.9) -> None:
        "initialize necessary  variables"
        self._train =True
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.random.normal(loc=0,scale=1,size=(1, num_features)) # initialize gamma randomly with 0 mean and unit std
        self.beta = np.random.normal(loc=0,scale=1,size=(1, num_features)) # initialize gamma randomly with 0 mean and unit std
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.zeros((1, num_features))
        self.dgamma = np.zeros((1, num_features))
        self.dbeta = np.zeros((1, num_features))
        self.mu = None
        self.sigma = None

    def forward(self,x : np.ndarray) -> np.ndarray:
        "Run forward pass"
        if  self._train: # if training mode use batch statistics
            self.x = x

            # calculate batch statistics
            self.mu =  np.mean(x,axis=0,keepdims=True)
            self.sigma =  np.var(x,axis=0,keepdims=True)
            self.xhot = (x - self.mu)/np.sqrt(self.sigma + self.eps)
            y = self.xhot * self.gamma + self.beta

            # calculate expenentially weighted average for test time
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.sigma
            return y
        else: # use running mean and variance in test time
            xhot = (x- self.running_mean)/np.sqrt(self.running_var + self.eps)
            y = xhot * self.gamma + self.beta
            return y

        
    def backward(self,grad : np.ndarray) -> np.ndarray:
        "Run backward pass"
        m = self.x.shape[0]
        dxhot = grad * self.gamma # x centered derivative
        dsigma = np.sum(dxhot * (self.x - self.mu) * (-1/2)* (self.sigma + self.eps)**(-3/2),axis=0, keepdims=True) # sigma derivative
        dmu = np.sum(dxhot * -1/np.sqrt(self.sigma + self.eps),axis=0,keepdims=True)  + dsigma * (-2 * (self.x - self.mu)).sum(axis=0, keepdims=True)/m # mu derivative
        dx = dxhot/np.sqrt(self.sigma + self.eps) + dsigma * 2 * (self.x - self.mu)/m + dmu/m # x derivative
        self.dgamma = np.sum(grad * self.xhot, axis=0, keepdims=True) # gamma derivative
        self.dbeta = np.sum(grad, axis=0, keepdims=True) # beta derivative
        return dx

    def __call__(self,x : np.ndarray):
        "Enable object to be called as a method"
        return self.forward(x)
    

    def update_parameters(self,updates: Tuple[np.ndarray] ) -> None:
        "Update parameters, Recieves updates"

        self.gamma -= updates[0]
        self.beta -= updates[1]

    def zero_grad(self) -> None:
        "Reset gradient"
        self.dgamma = np.zeros((1, self.num_features))
        self.dbeta = np.zeros((1, self.num_features))

    def get_derivatives(self) -> Tuple[np.ndarray]:
        "Return derivatives"
        return self.dgamma, self.dbeta
    
    def get_parameteters(self) ->  Tuple[np.ndarray]:
        "Return parameters"
        return self.gamma, self.beta
    
    def train(self):

        self._train = True
    def eval(self):

        self._train = False


class BacthNorm2d(Layer):

    def __init__(self,num_features : int, eps : float=1e-05, momentum: float=0.9) -> None:
        "initialize necessary  variables"
        self._train =True
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.random.normal(loc=0,scale=1,size=(1, num_features)) # initialize gamma randomly with 0 mean and unit std
        self.beta = np.random.normal(loc=0,scale=1,size=(1, num_features)) # initialize gamma randomly with 0 mean and unit std
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.zeros((1, num_features))
        self.dgamma = np.zeros((1, num_features))
        self.dbeta = np.zeros((1, num_features))
        self.mu = None
        self.sigma = None

    def forward(self,x : np.ndarray) -> np.ndarray:
        "Run forward pass"
        if  self._train: # if training mode use batch statistics
            self.x = x

            # calculate batch statistics
            self.mu =  np.mean(x,axis=(0,2,3))
            self.sigma =  np.var(x,axis=(0,2,3))

            self.xhot = (x - self.mu.reshape(1,-1,1,1))/np.sqrt(self.sigma.reshape(1,-1,1,1) + self.eps)
            y = self.xhot * self.gamma.reshape(1,-1,1,1) + self.beta.reshape(1,-1,1,1)

            # calculate expenentially weighted average for test time
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.sigma
            return y
        else: # use running mean and variance in test time
            xhot = (x- self.running_mean.reshape(1,-1,1,1))/np.sqrt(self.running_var.reshape(1,-1,1,1) + self.eps)
            y = xhot * self.gamma.reshape(1,-1,1,1) + self.beta.reshape(1,-1,1,1)
            return y

        
    def backward(self,grad : np.ndarray) -> np.ndarray:
        "Run backward pass"
        m = self.x.shape[0] * self.x.shape[-1] * self.x.shape[-2]
        dxhot = grad * self.gamma.reshape(1,-1,1,1) # x centered derivative
        dsigma = np.sum(dxhot * (self.x - self.mu.reshape(1,-1,1,1)) * (-1/2)* (self.sigma.reshape(1,-1,1,1) + self.eps)**(-3/2),axis=(0,2,3)) # sigma derivative

        dmu = np.sum(dxhot * -1/np.sqrt(self.sigma.reshape(1,-1,1,1) + self.eps),axis=(0,2,3))  + dsigma * (-2 * (self.x - self.mu.reshape(1,-1,1,1))).sum(axis=(0,2,3))/m # mu derivative
    
        dx = dxhot/np.sqrt(self.sigma.reshape(1,-1,1,1) + self.eps) + dsigma.reshape(1,-1,1,1) * 2 * (self.x - self.mu.reshape(1,-1,1,1))/m + dmu.reshape(1,-1,1,1)/m # x derivative
        self.dgamma = np.sum(grad * self.xhot, axis=(0,2,3)) # gamma derivative
        self.dbeta = np.sum(grad, axis=(0,2,3)) # beta derivative
        return dx

    def __call__(self,x : np.ndarray):
        "Enable object to be called as a method"
        return self.forward(x)
    

    def update_parameters(self,updates: Tuple[np.ndarray] ) -> None:
        "Update parameters, Recieves updates"

        self.gamma -= updates[0]
        self.beta -= updates[1]

    def zero_grad(self) -> None:
        "Reset gradient"
        self.dgamma = np.zeros((1, self.num_features))
        self.dbeta = np.zeros((1, self.num_features))

    def get_derivatives(self) -> Tuple[np.ndarray]:
        "Return derivatives"
        return self.dgamma, self.dbeta
    
    def get_parameteters(self) ->  Tuple[np.ndarray]:
        "Return parameters"
        return self.gamma, self.beta
    
    def train(self):

        self._train = True
    def eval(self):

        self._train = False
