from optimizers.optimizers import SGD, Adam, RMSProp, SGDMomentum
from losses.cross_entropy import CrossEntropy
from dataset.dataset import Dataset, Dataloader, MNISTDisgits
from layers.linear import Linear
from layers.batchnorm import BacthNorm1d
from layers.dropout import Dropout
from activations.activations import ReLU
from utils.initialization import xavier_initialization
from misc import accuracy, plot_metrics, plot_random_results

import numpy as np
import matplotlib.pyplot as plt


#Define Model
class MNISTModel:
    
    def __init__(self) -> None:
        "Initialize all layers to be used a model"

        self._train = True
        # initialize layers
        self.linear1 = Linear(784,1024,xavier_initialization)
        self.b1 = BacthNorm1d(1024)
        self.relu1 = ReLU()
        self.linear2 = Linear(1024,128,xavier_initialization)
        self.b2 = BacthNorm1d(128)
        self.relu2 = ReLU()
        self.linear3 = Linear(128,10,xavier_initialization)

    def forward(self,x: np.ndarray) -> np.ndarray:
        "Recieve flatten image and return logits"
        # Perform forward propagation
        x = self.linear1(x)
        x = self.b1(x,self.train)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.b2(x,self.train)
        x = self.relu2(x)
        x = self.linear3(x)
        return x
    
    def train(self):
        self._train = True
    def eval(self):
        self._train = False
    
    def backward(self,grad: np.ndarray) -> None:

        "Recieve derivative from loss function and perform back propagation"
        # Perform backward propagation
        grad = self.linear3.backward(grad)
        grad = self.relu2.backward(grad)
        grad = self.b2.backward(grad)
        grad = self.linear2.backward(grad)
        grad = self.relu1.backward(grad)
        grad = self.b1.backward(grad)
        grad = self.linear1.backward(grad)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        "Enable object to be called as a method"
        return self.forward(x)
    


model = MNISTModel()

# optimizer = SGD(model, learning_rate=0.01)
# optimizer = RMSProp(model, learning_rate=0.01)
# optimizer = SGDMomentum(model, learning_rate=0.01)
optimizer = Adam(model, learning_rate=0.01)

loss = CrossEntropy()


print("======= Loading MNIST DATA ========")
mnistdataset_train = MNISTDisgits()
mnistdataset_test = MNISTDisgits(split="Test")
print("=======       DONE         ========")
batch_size = 64
dataloader_train = Dataloader(mnistdataset_train, batch_size=batch_size,shuffle=True)
dataloader_test = Dataloader(mnistdataset_train, batch_size=batch_size,shuffle=False)

x_rand,y_rand = next(dataloader_test)


# Training Loop
epochs = 100
losses = []
accs = []
losses_test = []
accs_test = []
for i in range(epochs):
    losses_t = []
    accs_t = []
    losses_tr = []
    accs_tr = []
    model.train()
    for x,y in dataloader_train:
        optimizer.zero_grad()
        logits = model(x)
        preds  = np.argmax(logits,axis=1)
        acc = accuracy(preds,y)
        l = loss(logits, y)
        grad = loss.backward()
        model.backward(grad)
        optimizer.step()
        losses_tr.append(l)
        accs_tr.append(acc)
    mean_l = np.mean(losses_tr)
    mean_acc= np.mean(accs_tr)
    losses.append(mean_l)
    accs.append(mean_acc)
    print(f"Epoch: {i+1}/{epochs}, Training Loss: {mean_l:.3f}, Train Accuracy: {mean_acc:.2f}")
    model.eval()
    for x,y in dataloader_test:
        logits = model(x)
        preds  = np.argmax(logits,axis=1)
        acc = accuracy(preds,y)
        l = loss(logits, y)
        losses_t.append(l)
        accs_t.append(acc)
    mean_l = np.mean(losses_t)
    mean_acc= np.mean(accs_t)
    losses_test.append(mean_l)
    accs_test.append(mean_acc)
    print(f"Epoch: {i+1}/{epochs}, Test Loss: {mean_l:.3f}, Test Accuracy: {mean_acc:.2f}")

plot_metrics(losses,losses_test, "Train", "Test","Loss")
plot_metrics(accs,accs_test, "Train", "Test","Loss")

logits = model(x_rand)
preds  =  np.argmax(logits,axis=1)
plot_random_results(x_rand,preds,y_rand)