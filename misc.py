import matplotlib.pyplot as plt
import numpy as np

def accuracy(preds, labels):
    return (preds == labels).sum()/preds.shape[0]

def plot_metrics(train_metric,test_metric, label1, label2, title):
    plt.plot(range(len(train_metric)), train_metric)
    plt.plot(range(len(test_metric)), test_metric)
    plt.title(f"{title}")
    plt.legend([label1,label2])
    plt.show()


def plot_random_results(x,preds,labels,n=9):
    n_grids = int(np.ceil(np.sqrt(n)))
    plt.figure(figsize=(15,15))

    plt.subplots(n_grids,n_grids)

    for i in range(n):
        plt.subplot(n_grids,n_grids,i+1)
        plt.imshow(x[i].reshape(28,-1))
        plt.title("Predicted Label: {}".format(labels[i],preds[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()
    

