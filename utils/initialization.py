import numpy as np

def xavier_initialization(input,output):
    std = np.sqrt(2/(input + output))
    w = np.random.normal(scale=std, loc=0,size=(input,output))
    b = np.random.normal(scale=std, loc=0,size=(output,))
    return w, b