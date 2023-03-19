

import numpy as np

class CrossEntropy:

    def __call__(self,x: np.ndarray,labels: np.ndarray) -> np.ndarray:
        "Enable object to be called as a method"
        return self.forward(x,labels)
    
    def forward(self,logits: np.ndarray, labels: np.ndarray) -> float:
        "Calculate Cross Entropy loss using logsum trick"

        n_class = logits.shape[1]
        self.labels_onehot = self.one_hot_encoding(labels,n_class)
        self.probs = self.softmax(logits)
        c = np.max(logits,axis=1,keepdims=True)
        self.log_probs = logits - np.log(np.exp(logits - c).sum(axis=1,keepdims=True)) - c
        loss = -np.sum(self.labels_onehot * self.log_probs)/self.labels_onehot.shape[0]
        return loss

    def backward(self) -> None:
        "Calculate derivative"
        return self.probs - self.labels_onehot

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        "Calculate Softmax using exponential trick"

        exps = np.exp(x - np.max(x,axis=1,keepdims=True))
        return exps/np.sum(exps,axis=1, keepdims=True)
    
    @staticmethod
    def one_hot_encoding(labels: np.ndarray, n_class: np.ndarray) -> np.ndarray:
        "Take labels and return one hot encoding"
        labels_onehot = np.zeros(shape=(labels.shape[0], n_class))
        labels_onehot[np.arange(labels.size),labels] = 1
        return labels_onehot
