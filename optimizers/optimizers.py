
class SGD:
    "Stochastic Gradient Descent"
    def __init__(self, model : object , learning_rate : float=0.001) -> None:
        "Set learning rate and model to optimized"
        self.model = model
        self.learning_rate = learning_rate

    def step(self) -> None:
        "Update trainable parameters"
        attributes = vars(self.model)
        for att in attributes:
            layer = attributes[att]
            if layer.has_parameters():
                layer.update_parameters(self.learning_rate)

    def zero_grad(self) -> None:
        "Set gradient to zero"
        attributes = vars(self.model)
        for att in attributes:
            layer = attributes[att]
            if layer.has_parameters():
                layer.zero_grad()


