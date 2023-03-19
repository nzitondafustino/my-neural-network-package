
import numpy as np 
import pandas as pd
import random
from typing import Tuple

class Dataset:
    def __init__(self, x : np.ndarray=None, y : np.ndarray=None) -> None:
        "initialize data"
        self.data = list(zip(x, y))
    
    def __getitem__(self,index) -> Tuple:
        "Get single data instance"
        return self.data[index]
    
    def __len__(self) -> int:
        "Return length of dataset"
        return len(self.data)
    
    def shuffle(self) -> None:
        "Enable Shuffle functionality"
        random.shuffle(self.data)

    

class Dataloader:

    def __init__(self, dataset : Dataset, batch_size : int=64, collate_fn : object=None, shuffle : bool=True) -> None:
        "Initialize all necessary parameters"

        self.dataset = dataset
        self.batch_size = batch_size
        self.index = 0
        self.collate_df = collate_fn
        self.__is_shuffled = False
        self.shuffle = shuffle

    def __iter__(self) -> object:
        "Make Dataloader object iterable"
        self.index = 0
        return self
    
    def get(self) -> None:
        "Get single instance of data"
        item = self.dataset[self.index]
        self.index +=1
        return item

    
    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        "Return bachtch in every Iteration"
        if self.shuffle and not self.__is_shuffled:
            self.dataset.shuffle()
            self.__is_shuffled = True
        if self.index >= len(self.dataset):
            raise StopIteration
        batch_size = min(len(self.dataset) - self.index, self.batch_size)
        if batch_size < self.batch_size:
            self.__is_shuffled = False
        data = [self.get() for _ in range(batch_size)]

        if self.collate_df is not None:
            return self.collate_df(data)
        return self.default_collate_df(data)
    
    @staticmethod
    def default_collate_df(batch: Tuple):
        "Collate function to arganize data in way that's compatible with model"
        X = np.array([x for x,_ in batch])/256 # normalize pixels
        y = np.array([y for _,y in batch])

        return X,y
    



class MNISTDisgits(Dataset):
    def __init__(self, split="Train") -> None:
        "Customize dataset class to read MNIST data"
        data = pd.read_csv("dataset/data/mnist_784_csv.csv",delimiter=",")
        if split == "Train":
            data = data.loc[:6000, :]
        elif split == "Test":
            data = data.loc[6000:, :]
        else:
            raise Exception("split should be either train or Test")
        x = data.iloc[:,0:784].to_numpy()
        y = data["class"].to_numpy()
        self.data = list(zip(x, y))
        
        
        