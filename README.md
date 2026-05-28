# Neural Network Framework from Scratch

A deep learning framework built from the ground up in pure NumPy — no PyTorch, no TensorFlow. Designed to help learners understand exactly how modern neural networks work under the hood, from forward propagation to backpropagation, convolutions to batch normalization.

> **Why build this?** Popular frameworks abstract away the math. This project removes that abstraction entirely — every layer, every gradient, every optimizer is implemented from first principles so you can see *why* deep learning works, not just *that* it works.

---

## Features

Implements the full stack of a modern deep learning framework:

**Layers**
- Fully connected (Linear)
- 2D Convolution (Conv2d)
- Max Pooling (MaxPooling2d)
- Batch Normalization (BatchNorm1d, BatchNorm2d)
- Dropout
- Flatten

**Activations**
- ReLU, Sigmoid, Softmax

**Loss Functions**
- Cross-Entropy Loss (with forward and backward pass)

**Optimizers**
- SGD
- SGD with Momentum
- RMSProp
- Adam

**Utilities**
- Xavier weight initialization
- Dataset and Dataloader abstractions
- MNIST loader with train/test splits
- Accuracy metric, loss/accuracy plotting, prediction visualization

---

## Architecture

The framework mirrors the design patterns of PyTorch:

```
BaseModel          — abstract base class with forward(), backward(), train(), eval()
├── layers/        — Dense, Conv2d, BatchNorm, Dropout, Pooling, Flatten
├── activations/   — ReLU, Sigmoid, Softmax
├── losses/        — CrossEntropy
├── optimizers/    — SGD, SGDMomentum, RMSProp, Adam
├── dataset/       — Dataset, Dataloader, MNISTDigits
└── utils/         — Xavier initialization, plotting helpers
```

Every component implements its own `forward()` and `backward()` method, enabling automatic gradient flow through the computational graph via manual chain rule.

---

## Quick Start

### Installation

```bash
git clone https://github.com/nzitondafustino/my-neural-network-package.git
cd my-neural-network-package
pip install numpy matplotlib
```

### Define a Model

```python
from model import BaseModel
from layers.linear import Linear
from layers.batchnorm import BacthNorm1d
from activations.activations import ReLU
from utils.initialization import xavier_initialization
import numpy as np

class MLP(BaseModel):
    def __init__(self):
        self.linear1 = Linear(784, 256, xavier_initialization)
        self.bn1 = BacthNorm1d(256)
        self.relu1 = ReLU()
        self.linear2 = Linear(256, 10, xavier_initialization)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        return self.linear2(x)

    def backward(self, grad):
        grad = self.linear2.backward(grad)
        grad = self.relu1.backward(grad)
        grad = self.bn1.backward(grad)
        grad = self.linear1.backward(grad)
```

### Train on MNIST (CNN)

See `train.py` for a complete CNN training example that achieves strong accuracy on MNIST using:
- 2x Conv2d + BatchNorm2d + ReLU + MaxPooling blocks
- Fully connected head with BatchNorm1d
- Adam optimizer
- Cross-entropy loss
- Train/test evaluation loop with loss and accuracy tracking

```bash
python train.py
```

---

## Example: CNN Architecture (train.py)

```
Input (1x28x28)
  → Conv2d(1→64, kernel=5) + BatchNorm2d + ReLU + MaxPool(2)
  → Conv2d(64→128, kernel=5) + BatchNorm2d + ReLU + MaxPool(2)
  → Flatten → Linear(2048→128) + BatchNorm1d + ReLU
  → Linear(128→10)
  → CrossEntropy Loss
```

---

## What You Will Learn

By reading and running this codebase, you will understand:

- How **backpropagation** flows gradients through a computation graph
- How **convolutions** work at the matrix level, including gradient computation
- Why **batch normalization** stabilizes training and how its gradients are derived
- How **Adam, RMSProp, and SGD with momentum** differ in how they update weights
- How **Xavier initialization** prevents vanishing/exploding gradients
- How a **DataLoader** batches and shuffles data during training

---

## Roadmap

- [ ] Recurrent layers (RNN, LSTM)
- [ ] Additional loss functions (MSE, Binary Cross-Entropy)
- [ ] Learning rate schedulers
- [ ] Model save/load
- [ ] More datasets (CIFAR-10)

---

## Author

**Faustin Nzitonda**
MS ECE (Carnegie Mellon) | MSBA (Emory)
[github.com/nzitondafustino](https://github.com/nzitondafustino) | [linkedin.com/in/faustinnzitonda](https://linkedin.com/in/faustinnzitonda)

