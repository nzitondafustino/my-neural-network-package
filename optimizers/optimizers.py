import numpy as np

class Optimizer:
    "The Base optimizer class"
    def zero_grad(self) -> None:
        "Set gradient to zero"
        attributes = vars(self.model)
        for att in attributes:
            layer = attributes[att]
            if layer.has_parameters():
                layer.zero_grad()

    def step(self):
        raise NotImplemented
        


class SGD(Optimizer):
    "Stochastic Gradient Descent"
    def __init__(self, model : object , learning_rate : float=0.001) -> None:
        "Set learning rate and model to optimized"
        self.model = model
        self.learning_rate = learning_rate

    def step(self) -> None:
        "Update trainable parameters using SGD"
        attributes = vars(self.model)
        for att in attributes:
            layer = attributes[att]
            if layer.has_parameters():
                derivatives =  tuple([self.learning_rate * derivative  for derivative in layer.get_derivatives()])
                layer.update_parameters(derivatives)

class SGDMomentum(Optimizer):
    "Stochastic Gradient Descent with momentum"
    def __init__(self, model : object , learning_rate : float=0.001, beta : float = 0.9) -> None:
        "Set learning rate and model to optimized"
        self.model = model
        self.learning_rate = learning_rate
        self.beta = beta
        self.m = {}

    def step(self) -> None:
        "Update trainable parameters using SGD with momentum"
        attributes = vars(self.model)
        for att in attributes:
            layer = attributes[att]
            if layer.has_parameters():
                derivatives =  layer.get_derivatives() # get derivatives from layer
                if id(layer) not in self.m:
                    self.m[id(layer)] = [0] * len(derivatives)
                updates = [0] * len(derivatives)

                for i, derivative in enumerate(derivatives):

                    # calculate moving avarage
                    mt = self.beta * self.m[id(layer)][i] + self.learning_rate * derivative

                    # update moving avarage
                    self.m[id(layer)][i] = mt

                    updates[i] = self.learning_rate * mt
                #update parameters
                layer.update_parameters(updates)



class RMSProp(Optimizer):
    "RMSProp optimizer"
    def __init__(self, model : object, learning_rate :float = 0.001, beta : float = 0.9, epsilon : float = 10e-8) -> None:
        "Initialize necessary parameters for RMSProp Optimizer"
        self.beta = beta
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.v = {}
        self.model = model

    def step(self) -> None:
        "Update trainable parameters using RMSProp"
        attributes = vars(self.model)
        for att in attributes:
            layer = attributes[att]
            if layer.has_parameters():
                derivatives =  layer.get_derivatives() # get derivatives from layer

                # initialize previous first moments for the firt derivative
                if id(layer) not in self.v:
                    self.v[id(layer)] = [0] * len(derivatives)

                updates = [0] * len(derivatives)
                for i, derivative in enumerate(derivatives):
                    # calculate moving avarage
                    vt = self.beta * self.v[id(layer)][i] + (1 -  self.beta) * derivative ** 2

                    # update calculate moving avarage
                    self.v[id(layer)][i] = vt

                    updates[i] = self.learning_rate * derivative/np.sqrt( vt + self.epsilon)
                #update parameters
                layer.update_parameters(updates)
        
    
class Adam(Optimizer):
    "Adam Optimizer"
    def __init__(self, model : object, learning_rate : float = 0.001, beta1 : float = 0.9, beta2 : float = 0.999, epsilon : float = 10e-8) -> None:
        "Initialize necessary parameters for Adam Optimizer"
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.t = 0
        self.m = {}
        self.v = {}
        self.model = model

    def step(self):
        "Update trainable parameters using Adam"
        self.t += 1
        attributes = vars(self.model)
        for att in attributes:
            layer = attributes[att]
            if layer.has_parameters():
                derivatives =  layer.get_derivatives() # get derivatives from layer

                # initialize previous first and second moments for the firt derivative
                if id(layer) not in self.m:
                    self.m[id(layer)] = [0] * len(derivatives)
                    self.v[id(layer)] = [0] * len(derivatives)

                updates = [0] * len(derivatives)
                for i, derivative in enumerate(derivatives):
                    # calculate moving avarages
                    mt = self.beta1 * self.m[id(layer)][i] + (1 -  self.beta1) * derivative
                    vt = self.beta2 * self.v[id(layer)][i] + (1 -  self.beta2) * derivative ** 2

                    # update moving avarages
                    self.m[id(layer)][i] = mt
                    self.v[id(layer)][i] = vt

                    # correct biases

                    mt_hot = mt/(1 - self.beta1**self.t)
                    vt_hot = vt/(1 - self.beta2**self.t)

                    updates[i] = self.learning_rate * mt_hot/np.sqrt( vt_hot + self.epsilon)
                #update parameters
                layer.update_parameters(updates)
        









