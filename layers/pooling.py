from typing import Any
from layers.linear import Layer
import numpy as np


class MaxPooling2d(Layer):
    def __init__(self, kernel_size, stride = None, padding=0) -> None:
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size
        self.padding = padding

        if isinstance(padding, int):
            self.padding = tuple([padding]*2)

        if isinstance(self.stride, int):

            self.stride = tuple([self.stride]*2)

        if isinstance(kernel_size, int):
            self.kernel_size = tuple([kernel_size]*2)

    def forward(self, x: np.ndarray) -> np.ndarray:
        b,c,hin,win = x.shape
        hout =(hin + 2 * self.padding[0] - self.kernel_size[0])//self.stride[0] + 1
        wout =(win + 2 * self.padding[1] - self.kernel_size[0])//self.stride[1] + 1

        out = np.zeros((b,c, hout, wout))
        
        if self.padding[0] or self.padding[1]:
            self.x = np.zeros((b,c,2*self.padding[0] + hin, 2*self.padding[1] + win))
            self.x[:,:, self.padding[0]:-self.padding[0],self.padding[1]:-self.padding[1]] = x
        else:
            self.x = x

        self.max_indeces = np.zeros_like(self.x)
        
        for h in range(hout):
            start_h = h*self.stride[0]
            end_h =  start_h + self.kernel_size[0]
            for w in range(wout):
                start_w = w*self.stride[1]
                end_w =  start_w + self.kernel_size[1]
                window = self.x[:,:,start_h:end_h, start_w:end_w]

                argmax1 = np.argmax(window, axis=-1)
                max1 = np.take_along_axis(window, np.expand_dims(argmax1, axis=-1), axis=-1).squeeze(axis=-1)

                argmax2 = np.argmax(max1, axis=-1)
                max2 = np.take_along_axis(max1, np.expand_dims(argmax2, axis=-1), axis=-1).squeeze(axis=-1)
                tmp = np.zeros_like(argmax1)
                np.put_along_axis(tmp,np.expand_dims(argmax2, axis=-1),1, axis=-1)
                np.put_along_axis(self.max_indeces[:,:,start_h:end_h, start_w:end_w],np.expand_dims(argmax1, axis=-1),np.expand_dims(tmp, axis=-1),axis=-1)

                out[:,:,h,w] = max2
        return out
    
    def has_parameters(self) -> bool:
        "Return True if a layer is trainable"
        return False
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        b,c,hin,win = grad.shape
        new_grad = np.zeros_like(self.x, dtype=np.float32)
        for h in range(hin):
            start_h = h*self.stride[0]
            end_h =  start_h + self.kernel_size[0]
            for w in range(win):
                start_w = w*self.stride[1]
                end_w =  start_w + self.kernel_size[1]
                new_grad[:,:,start_h:end_h, start_w:end_w] += (self.max_indeces[:,:,start_h:end_h, start_w:end_w]\
                                                              .reshape(b*c,-1) * grad[:,:,h,w].reshape(b*c,-1))\
                                                                .reshape(b,c,self.kernel_size[0], self.kernel_size[1])
        if self.padding[0] or self.padding[1]:
            return new_grad[:,:, self.padding[0]:-self.padding[0],self.padding[1]:-self.padding[1]]
        
        return new_grad
        
class MeanPooling2d(Layer): 

    def __init__(self, kernel_size, stride = 1, padding=0) -> None:
        self.kernel_size = kernel_size
        self.stride = stride 
        self.padding = padding

        if isinstance(padding, int):
            self.padding = tuple([padding]*2)

        if isinstance(stride, int):

            self.stride = tuple([stride]*2)

        if isinstance(kernel_size, int):
            self.kernel_size = tuple([kernel_size]*2)

    def forward(self, x: np.ndarray) -> np.ndarray:
        b,c,hin,win = x.shape
        hout =(hin + 2 * self.padding[0] - self.kernel_size[0])//self.stride[0] + 1
        wout =(win + 2 * self.padding[1] - self.kernel_size[0])//self.stride[1] + 1
        out = np.zeros((b,c, hout, wout))

        if self.padding[0] or self.padding[1]:
            self.x = np.zeros((b,c,2*self.padding[0] + hin, 2*self.padding[1] + win))
            self.x[:,:, self.padding[0]:-self.padding[0],self.padding[1]:-self.padding[1]] = x
        else:
            self.x = x

        for h in range(hout):
            start_h = h*self.stride[0]
            end_h =  start_h + self.kernel_size[0]
            for w in range(wout):
                start_w = w*self.stride[1]
                end_w =  start_w + self.kernel_size[1]
                window = self.x[:,:,start_h:end_h, start_w:end_w]
                out[:,:,h,w] = window.mean(axis=(2,3))

        return out
    
    def has_parameters(self) -> bool:
        "Return True if a layer is trainable"
        return False
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)
    
    def backward(self, grad: np.ndarray) -> np.ndarray:

        b,c,hin,win = grad.shape
        new_grad = np.zeros_like(self.x, dtype=np.float32)
        for h in range(hin):
            start_h = h*self.stride[0]
            end_h =  start_h + self.kernel_size[0]
            for w in range(win):
                start_w = w*self.stride[1]
                end_w =  start_w + self.kernel_size[1]
                new_grad[:,:,start_h:end_h, start_w:end_w] += (1/(self.kernel_size[0] * self.kernel_size[1]) * grad[:,:,h,w]).reshape(b,c,1,1)
        
        if self.padding[0] or self.padding[1]:
            return new_grad[:,:, self.padding[0]:-self.padding[0],self.padding[1]:-self.padding[1]]
        
        return new_grad

class MaxPooling1d(Layer):

    def __init__(self, kernel_size, stride = None, padding=0) -> None:
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size
        self.padding = padding

    def forward(self, x: np.ndarray) -> np.ndarray:
        b,c,hin = x.shape
        hout =(hin + 2 * self.padding - self.kernel_size)//self.stride + 1
        out = np.zeros((b,c, hout))
        
        if self.padding:
            self.x = np.zeros((b,c,2*self.padding + hin))
            self.x[:,:, self.padding:-self.padding] = x
        else:
            self.x = x

        self.max_indeces = np.zeros_like(self.x)

        for h in range(hout):
            start_h = h*self.stride
            end_h =  start_h + self.kernel_size
            window = self.x[:,:,start_h:end_h]

            argmax1 = np.argmax(window, axis=-1)
            max1 = np.take_along_axis(window, np.expand_dims(argmax1, axis=-1), axis=-1).squeeze(axis=-1)
            np.put_along_axis(self.max_indeces[:,:,start_h:end_h],np.expand_dims(argmax1, axis=-1),1,axis=-1)
            out[:,:,h] = max1

        return out
    
    def has_parameters(self) -> bool:
        "Return True if a layer is trainable"
        return False
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)
    
    def backward(self, grad: np.ndarray) -> np.ndarray:

        b,c,hin = grad.shape
        new_grad = np.zeros_like(self.x)
        for h in range(hin):
            start_h = h*self.stride
            end_h =  start_h + self.kernel_size
            new_grad[:,:,start_h:end_h] += (self.max_indeces[:,:,start_h:end_h]\
                                                            .reshape(b*c,-1) * grad[:,:,h].reshape(b*c,-1))\
                                                            .reshape(b,c,self.kernel_size)
            
        if self.padding:
            return new_grad[:,:, self.padding:-self.padding]
        return new_grad


class MeanPooling1d(Layer):

    def __init__(self, kernel_size, stride = 1, padding=0) -> None:
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: np.ndarray) -> np.ndarray:
        b,c,hin = x.shape
        hout =(hin + 2 * self.padding - self.kernel_size)//self.stride + 1
        out = np.zeros((b,c, hout))
        self.max_indeces = np.zeros_like(x)
        
        if self.padding:
            self.x = np.zeros((b,c,2*self.padding + hin))
            self.x[:,:, self.padding:-self.padding] = x
        else:
            self.x = x
        
        for h in range(hout):
            start_h = h*self.stride
            end_h =  start_h + self.kernel_size
            window = self.x[:,:,start_h:end_h]
            out[:,:,h] = window.mean(axis=(2))
        return out
    
    def has_parameters(self) -> bool:
        "Return True if a layer is trainable"
        return False
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)
    
    def backward(self, grad: np.ndarray) -> np.ndarray:

        b,c,hin = grad.shape
        new_grad = np.zeros_like(self.x,dtype=np.float32)
        for h in range(hin):
            start_h = h*self.stride
            end_h =  start_h + self.kernel_size
            new_grad[:,:,start_h:end_h] += (1/(self.kernel_size) * grad[:,:,h]).reshape(b,c,1)

        if self.padding:
            return new_grad[:,:, self.padding:-self.padding]
        return new_grad