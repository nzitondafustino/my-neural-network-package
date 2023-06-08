from layers.linear import Layer
import numpy as np
from typing import Any, Tuple, Union, List


class Flatten(Layer):

    def has_parameters(self) -> bool:
        "Return True if a layer is trainable"
        return False
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Flatten the input with B,C,H,W: into B, C*H*W size
        Parameters
        -----------
            x: numpy array
                multidimentional array to be flattened

        Returns
        -------
            Flattened numpy array
        """
        self.b, self.c, self.h, self.w = x.shape
        return x.reshape(self.b,-1)
    
    def backward(self, grad: np.ndarray) -> np.ndarray:

        """
        Change the flattened gradient B, C*H*W into B,C,H,W size
        Parameters
        -----------
            x: numpy array
                multidimentional array to be transformed into original values size

        Returns
        -------
            gradient with the same size ad the input
        """

        return grad.reshape(self.b, self.c, self.h, -1)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
    

class Conv2d(Layer):

    """
    Convolution 2d neural network blueprint

    Atributes
    ---------
    n_channels: int
    out_channels :int
    kernel_size: Tuple
    dilation : Tuple
    padding : Tuple
    padding_value: float

    Methods
    -------
    forward(self, x: np.ndarray)
    backward(self, grad: np.ndarray)
    update_parameters(self,updates: Tuple[np.ndarray])
    zero_grad(self)
    get_derivatives(self)
    get_parameteters(self)
    """
    def __init__(self, n_channels : int,
                out_channels : int, 
                kernel_size : int, 
                stride : Union[int, List, Tuple] = 1, 
                padding : Union[int, List,Tuple] = 0,
                padding_value: float = 0, 
                dilation: Union[int, List, Tuple] = 1) -> None:
        
        """Initialize necesary parameters for the Layer
        Paramters:
            n_channels: int
                number of input channels
            out_channels :int
                number of output channels
            kernel_size: int/list/tuple
                kernel size
            dilation : int/list/tuple
                filter dilation
            padding : int/list/tuple
                padding size
            padding_value: float
                padding value
        
        """
        self.n_channels = n_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.padding_value = padding_value
        
        if isinstance(padding, int): # if integer is provided as input make it a tuple
            self.padding = tuple([padding]*2)

            
        if isinstance(kernel_size, int): # if integer is provided as input make it a tuple
            self.kernel_size = tuple([kernel_size]*2)

        if isinstance(stride, int): # if integer is provided as input make it a tuple
            self.stride = tuple([stride]*2)
        
        #parameters initialization using xavier uniform
        k = 1/(self.n_channels *  np.sum(self.kernel_size)) 
        self.w = np.random.uniform(-np.sqrt(k), np.sqrt(k), size=(out_channels, n_channels, self.kernel_size[0], self.kernel_size[1]))
        self.b = np.random.uniform(-np.sqrt(k), np.sqrt(k), size=(out_channels))

        # initialized parameters gradients to zero
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)


    def __convolution(self, x, weight,bias, hout, wout, c_out, kernel_size,stride, axes=((3,2,1),(3,2,1))):
            
        """
        Perform convolution using tensordot
        """
        b,_,_,_ =  x.shape
        out = np.zeros((b,c_out, hout, wout))
        for h in range(hout):
            h_start = h * stride[0]
            h_end = h_start + kernel_size[0]
            for w in range(wout):
                w_start = w * stride[1]
                w_end = w_start + kernel_size[1]

                window = x[:,:,h_start:h_end, w_start:w_end]
                tmp = np.tensordot(window, weight, axes=axes) + bias
                out[:,:,h,w] += tmp

        return out
    
    def __parameters_gradient(self, grad):
        """
        Perform gradient of parameters of parameters
        """
        b,c,hin,win = grad.shape

        # create new size by padding dilating the incomming gradient
        newhin = hin  + (hin - 1) * (self.stride[0] - 1)
        newwin = win + (win - 1) * (self.stride[1] - 1)

        # dilate the new grad
        new_grad = np.zeros((b,c,newhin,newwin))
        new_grad[:,:,::self.stride[0],::self.stride[1]] = grad
        _,_,hout, wout = self.__dilate_filter().shape
        self.db = grad.sum(axis=(2,3)).mean(axis=0)

        # dilate filter
        dilated_filter = self.__dilate_filter()
        dw_dilated = np.zeros_like(dilated_filter)

        # perform convolution
        for h in range(hout):
            h_start = h
            h_end = h_start + newhin
            for w in range(wout):
                w_start = w
                w_end = w_start + newwin
                window = self.x[:,:,h_start:h_end, w_start:w_end]
                tmp = np.tensordot(np.mean(new_grad,axis=0), np.mean(window,axis=0), axes=((2,1),(2,1)))
                dw_dilated[:,:,h,w] = tmp 
        self.dw = dw_dilated[:,:,::self.dilation,::self.dilation]

    def __calculate_dx(self,grad):
        """
        Calculate derivative with respect to the input
        """

        # dilate filter
        dilated_filter = self.__dilate_filter()
        kernel_size = tuple([dilated_filter.shape[-1]]*2)

        # rotate filter by 180 degrees
        flipped_filter = np.rot90(dilated_filter,k=2, axes=(2,3))
        b,c,hin,win = grad.shape

        # pad input gradient by kernel size and insert zeros between elements(number of zeros equal to stride - 1)
        newhin = hin  + (hin - 1) * (self.stride[0] - 1) + 2 * (kernel_size[0] - 1)
        newwin = win + (win - 1) * (self.stride[1] - 1) + 2 * (kernel_size[1] - 1)
        new_grad = np.zeros((b,c,newhin,newwin))

        new_grad[:,:,kernel_size[0] - 1:-kernel_size[0] + 1:self.stride[0],kernel_size[1] - 1:-kernel_size[1] + 1:self.stride[1]] = grad
        _,_,hout, wout = self.x.shape
        dx = np.zeros_like(self.x)

        # perform convolution of gradeint with rotated filter[ conv(grad, 180 rotated filter)]
        for h in range(hout):
            h_start = h
            h_end = h_start + kernel_size[0]
            for w in range(wout):
                w_start = w
                w_end = w_start + kernel_size[1]
                window = new_grad[:,:,h_start:h_end, w_start:w_end]
                tmp = np.tensordot(window, flipped_filter, axes=((3,2,1),(3,2,0)))
                dx[:,:,h,w] = tmp
        if self.padding[0] or self.padding[1]:
            return dx[:,:,self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1]]
        return dx
    
    def __dilate_filter(self):
        "Dilate filter if dilation is different from 1"
        cout, cin, k, k = self.w.shape
        if self.dilation > 1:
            newk = k + (self.dilation - 1) * (k - 1)
            dilated_filter = np.zeros((cout, cin, newk, newk))
            dilated_filter[:,:,::self.dilation,::self.dilation] = self.w
        else:
            dilated_filter = self.w
        return dilated_filter

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Take input x(size of B,Cin,Hin,Win) and return the forward pass(size B,Cout, Hout, Wout)
        Parameters
        ----------
        x : numpy array
            input to be convolved

        Returns
        -------
        Numpy array: results of convolution
        """

        b,cin, hin, win =  x.shape

        # calculate output size
        hout = (hin + 2 * self.padding[0] - self.dilation  * (self.kernel_size[0] - 1) - 1)//self.stride[0] + 1
        wout = (win  + 2 * self.padding[1] - self.dilation  * (self.kernel_size[1] - 1) - 1)//self.stride[1] + 1

        # padd if passing is not zero
        if self.padding[0] or self.padding[1]:
            self.x = np.ones((b,cin,hin + 2 * self.padding[0], win + 2 * self.padding[1])) * self.padding_value
            self.x[:,:,self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1]] = x
        else:
            self.x = x
        
        # dilate filter
        dilated_filter = self.__dilate_filter()
        kernel_size = tuple([dilated_filter.shape[-1]]*2)

        out = self.__convolution(self.x, dilated_filter,self.b, hout, wout, self.out_channels,kernel_size, self.stride) # perform convolution
        return out
    
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Take input grad(size of B,Cout,Hout,Wout) and return the packward pass(size B,Cin, Hin, Win)
        Parameters
        ----------
        grad : numpy array
            gradient of the preceding layer

        Returns
        -------
            Numpy array: gradient to be propagated to the previous layer
        """
        # calculate derivative with respect to filter: conv(input,grad)
        self.__parameters_gradient(grad)
        # calculate derivative with respect to input: full conv(rotated 180 filter, grad)
        return self.__calculate_dx(grad)

    def __call__(self, x):
        return self.forward(x)
    

    def update_parameters(self,updates: Tuple[np.ndarray] ) -> None:
        "Update parameters, Recieves updates"

        self.w -= updates[0]
        self.b -= updates[1]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        "Enable object to be called as a method"
        return self.forward(x)

    def zero_grad(self) -> None:
        "Reset gradient"
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)

    def get_derivatives(self):
        "Return derivatives"
        return self.dw, self.db
    
    def get_parameteters(self):
        "Return parameters"
        return self.w, self.b
    


class Conv1d(Layer): 
    """
    Convolution 1d neural network blueprint: for reconstructing tensor using learnable parameters

    Atributes
    ---------
    n_channels: int
    out_channels :int
    kernel_size: int
    dilation : int
    padding : int
    padding_value: float

    Methods
    -------
    forward(self, x: np.ndarray)
    backward(self, grad: np.ndarray)
    update_parameters(self,updates: Tuple[np.ndarray])
    zero_grad(self)
    get_derivatives(self)
    get_parameteters(self)
    """

    def __init__(self, 
                 n_channels : int, 
                 out_channels : int, 
                 kernel_size : int, 
                 stride : int=1, 
                 padding : int=0,
                 padding_value : float=0, 
                 dilation : int=1) -> None:
        
        """Initialize necesary parameters for the Layer
        Paramters:
            n_channels: int
                number of input channels
            out_channels :int
                number of output channels
            kernel_size: int
                kernel size
            dilation : int
                filter dilation
            padding : int
                padding size
            padding_value: float
                padding value
        
        """

        self.n_channels = n_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.padding_value = padding_value
        self.stride = stride

        #initialize parameters using xavier normal

        k = 1/(self.n_channels *  self.kernel_size)
        self.w = np.random.uniform(-np.sqrt(k), np.sqrt(k), size=(out_channels, n_channels, self.kernel_size))
        self.b = np.random.uniform(-np.sqrt(k), np.sqrt(k), size=(out_channels))

        #initialize parameters to 0

        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)

    def __dilate_filter(self):
        "Dilate filter if dilation is different from 1"
        cout, cin, k = self.w.shape
        if self.dilation > 1:
            newk = k + (self.dilation - 1) * (k - 1)
            dilated_filter = np.zeros((cout, cin, newk))
            dilated_filter[:,:,::self.dilation] = self.w
        else:
            dilated_filter = self.w
        return dilated_filter
    
    def __convolution(self, x, weight,bias, hout, c_out, kernel_size,stride, axes=((2,1),(2,1))):
        """
        Perform convolution using tensordot
        """
        if len(x.shape) == 2:
            x = np.expand_dims(x,axis=0)
        b,_, _ = x.shape
        out = np.zeros((b,c_out, hout))
        for h in range(hout):
            h_start = h * stride
            h_end = h_start + kernel_size
            window = x[:,:,h_start:h_end]
            tmp = np.tensordot(window, weight, axes=axes) + bias
            out[:,:,h] += tmp
        return out
    
    def __parameters_gradient(self, grad):
        """
        Perform gradient of parameters of parameters
        """
        if len(grad.shape) == 2:
            grad = np.expand_dims(grad,axis=0)
        b,c, hin = self.x.shape
        newhin = hin  + (hin - 1) * (self.stride[0] - 1)
        new_grad = np.zeros((b,c,newhin))
        new_grad[:,:,::self.stride] = grad
        _,_,hout = self.__dilate_filter().shape
        self.db = grad.sum(axis=(2,)).mean(axis=0)

        dilated_filter = self.__dilate_filter()
        dw_dilated = np.zeros_like(dilated_filter)
        for h in range(hout):
            h_start = h
            h_end = h_start + newhin
            window = self.x[:,:,h_start:h_end]
            tmp = np.tensordot(np.mean(new_grad,axis=0), np.mean(window,axis=0), axes=((1),(1)))
            dw_dilated[:,:,h] = tmp 
        self.dw = dw_dilated[:,:,::self.dilation]
    
    def __calculate_dx(self,grad):
        """
        Calculate derivative with respect to the input
        """
        if len(grad.shape) == 2:
            grad = np.expand_dims(grad,axis=0)

        dilated_filter = self.__dilate_filter()
        kernel_size = dilated_filter.shape[-1]
        flipped_filter = dilated_filter[:,:,::-1]
        b,c,hin = grad.shape
        newhin = hin  + (hin - 1) * (self.stride - 1) + 2 * (kernel_size - 1)
        new_grad = np.zeros((b,c,newhin))
        new_grad[:,:,kernel_size - 1:-kernel_size + 1:self.stride] = grad
        _,_,hout= self.x.shape
        dx = np.zeros_like(self.x)

        for h in range(hout):
            h_start = h
            h_end = h_start + kernel_size
            window = new_grad[:,:,h_start:h_end]
            tmp = np.tensordot(window, flipped_filter, axes=((2,1),(2,0)))
            dx[:,:,h] = tmp
        if self.padding:
            return dx[:,:,self.padding:-self.padding]
        
        if dx.shape[0] == 1:
            return dx.squeeze(axis=0)
        
        return dx

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Take input x(size of B,Cin,Hin) and return the forward pass(size B,Cout, Hout)
        Parameters
        ----------
        x : numpy array
            input to be convolved

        Returns
        -------
        Numpy array: results of convolution
        """

        if len(x.shape) == 2:
            x = np.expand_dims(x,axis=0)
        b,cin, hin = x.shape

        hout = (hin  + 2 * self.padding - self.dilation  * (self.kernel_size - 1) - 1)//self.stride + 1

        if self.padding:
            self.x = np.ones((b,cin,hin + 2 * self.padding)) * self.padding_value
            self.x[:,:,self.padding:-self.padding] = x
        else:
            self.x = x
        dilated_filter = self.__dilate_filter()
        kernel_size = dilated_filter.shape[-1]
        out = self.__convolution(self.x, dilated_filter,self.b, hout, self.out_channels,kernel_size, self.stride)
        if out.shape[0] == 1:
            return out.squeeze(axis=0)
        
        return out

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)
    
    def backward(self, grad: np.ndarray) -> np.ndarray:

        """
        Take input grad(size of B,Cout,Hout) and return the packward pass(size B,Cin, Hin)
        Parameters
        ----------
        grad : numpy array
            gradient of the preceding layer

        Returns
        -------
            Numpy array: gradient to be propagated to the previous layer
        """
        
        # calculate derivative with respect to filter: conv(input,grad)
        self.__parameters_gradient(grad)
        # calculate derivative with respect to input: full conv(rotated 180 filter, grad)
        return self.__calculate_dx(grad)
       
    def update_parameters(self,updates: Tuple[np.ndarray] ) -> None:
        "Update parameters, Recieves updates"

        self.w -= updates[0]
        self.b -= updates[1]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        "Enable object to be called as a method"
        return self.forward(x)

    def zero_grad(self) -> None:
        "Reset gradient"
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)

    def get_derivatives(self):
        "Return derivatives"
        return self.dw, self.db
    
    def get_parameteters(self):
        "Return parameters"
        return self.w, self.b


class ConvTransposed1d(Layer):

    """
    Transposed Convolution 1d neural network blueprint: for reconstructing tensor using learnable parameters

    Atributes
    ---------
    n_channels: int
    out_channels :int
    kernel_size: int
    dilation : int
    padding : int
    padding_value: float

    Methods
    -------
    forward(self, x: np.ndarray)
    backward(self, grad: np.ndarray)
    update_parameters(self,updates: Tuple[np.ndarray])
    zero_grad(self)
    get_derivatives(self)
    get_parameteters(self)
    """

    def __init__(self, 
                 n_channels : int, 
                 out_channels : int, 
                 kernel_size : int, 
                 stride : int=1, 
                 padding : int=0,
                 padding_value : float=0, 
                 dilation : int=1, 
                 output_padding : int=0) -> None:
        
        """Initialize necesary parameters for the Layer
        Paramters:
            n_channels: int
                number of input channels
            out_channels :int
                number of output channels
            kernel_size: int
                kernel size
            dilation : int
                filter dilation
            padding : int
                padding size
            padding_value: float
                padding value
        
        """

        self.n_channels = n_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.padding_value = padding_value
        self.stride = stride
        self.output_padding = output_padding
        
        #initialize parameters using xavier normal

        k = 1/(self.out_channels *  self.kernel_size)
        self.w = np.random.uniform(-np.sqrt(k), np.sqrt(k), size=(n_channels, out_channels, self.kernel_size))
        self.b = np.random.uniform(-np.sqrt(k), np.sqrt(k), size=(out_channels))

        #initialize parameters to 0
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)

    def __convolution(self, x, weight,bias, hout, c_out, kernel_size,stride, axes=((2,1),(2,0))):
        """
        Perform convolution using tensordot
        """
        b,_,_,=  x.shape
        out = np.zeros((b,c_out, hout))
        for h in range(hout):
            h_start = h * stride
            h_end = h_start + kernel_size
            window = x[:,:,h_start:h_end]
            tmp = np.tensordot(window, weight, axes=axes) + bias
            out[:,:,h] += tmp
        return out
    
    def __calculate_dx(self,grad):
        """
        Calculate derivative with respect to the input
        """
        dilated_filter = self.__dilate_filter()
        kernel_size = dilated_filter.shape[-1]
        b,c,hin = grad.shape
        newhin = hin + 2 * (kernel_size - 1)
        new_grad = np.zeros((b,c,newhin))
        new_grad[:,:,kernel_size - 1:-kernel_size + 1] = grad
        _,_,hout = self.x.shape
        dx = np.zeros_like(self.x)

        for h in range(hout):
            h_start = h
            h_end = h_start + kernel_size
            window = new_grad[:,:,h_start:h_end]
            tmp = np.tensordot(window, dilated_filter, axes=((2,1),(2,1)))
            dx[:,:,h] = tmp

        if self.new_padding:
            return dx[:,:, self.new_padding:-self.new_padding:self.stride]
        return dx


    def __dilate_filter(self):

        "Dilate filter if dilation is different from 1"
        cout, cin, k = self.w.shape
        if self.dilation > 1:
            newk = k + (self.dilation - 1) * (k - 1)
            dilated_filter = np.zeros((cout, cin, newk))
            dilated_filter[:,:,::self.dilation] = self.w
        else:
            dilated_filter = self.w
        return dilated_filter
    
    def __parameters_gradient(self, grad):
        """
        Perform gradient of parameters of parameters
        """
        _,_,hin = grad.shape
        _,_,hout = self.__dilate_filter().shape
        self.db = grad.sum(axis=(2)).mean(axis=0)

        dilated_filter = self.__dilate_filter()
        dw_dilated = np.zeros_like(dilated_filter)
        for h in range(hout):
            h_start = h
            h_end = h_start + hin
            window = self.x[:,:,h_start:h_end]
            tmp = np.tensordot(np.mean(window,axis=0), np.mean(grad,axis=0), axes=((1),(1)))
            dw_dilated[:,:,h] = tmp 
        self.dw = dw_dilated[:,:,::self.stride]

    def forward(self, x: np.ndarray) -> np.ndarray:

        """
        Take input x(size of B,Cin,Hin) and return the forward pass(size B,Cout, Hout)
        Parameters
        ----------
        x : numpy array
            input to be convolved

        Returns
        -------
        Numpy array: results of convolution
        """

        #Note: this is the same as conv1d except a few changes, stride dilates input  and the input is padding by kernel size - 2. paddings behave the same as othe convolutions
        b,cin, hin =  x.shape
        hout = (hin - 1) * self.stride - 2 * self.padding + self.dilation * (self.kernel_size - 1)  + self.output_padding + 1
        
        dilated_filter = self.__dilate_filter()
        kernel_size = dilated_filter.shape[-1]
        dilation = self.stride - 1 # calculate dilation size
        self.new_padding = (kernel_size - self.padding - 1)  # caluculate new padding
        self.x = np.zeros((b,cin, hin + 2 * self.new_padding + self.output_padding + (hin - 1) * (self.stride - 1)))

        self.x[:,:, self.new_padding:-self.new_padding:dilation + 1] = x
        stride = 1

        # filter is rotated by 180 degrees
        flipped_filter = dilated_filter[:,:,::-1]
        out = self.__convolution(self.x, flipped_filter,self.b, hout, self.out_channels, kernel_size,stride)  # perform convolution   
        return out
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Take input grad(size of B,Cout,Hout) and return the packward pass(size B,Cin, Hin)
        Parameters
        ----------
        grad : numpy array
            gradient of the preceding layer

        Returns
        -------
            Numpy array: gradient to be propagated to the previous layer
        """
        # calculate derivative with respect to filter: conv(input,grad)
        self.__parameters_gradient(grad)
        # calculate derivative with respect to input: full conv(rotated 180 filter, grad)
        return self.__calculate_dx(grad)
    

    def update_parameters(self,updates: Tuple[np.ndarray] ) -> None:
        "Update parameters, Recieves updates"

        self.w -= updates[0]
        self.b -= updates[1]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        "Enable object to be called as a method"
        return self.forward(x)

    def zero_grad(self) -> None:
        "Reset gradient"
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)

    def get_derivatives(self):
        "Return derivatives"
        return self.dw, self.db
    
    def get_parameteters(self):
        "Return parameters"
        return self.w, self.b



class ConvTransposed2d(Layer):

    """
    Transposed Convolution 2d neural network blueprint: for reconstructing tensor using learnable parameters

    Atributes
    ---------
    n_channels: int
    out_channels :int
    kernel_size: Tuple
    dilation : Tuple
    padding : Tuple
    padding_value: float

    Methods
    -------
    forward(self, x: np.ndarray)
    backward(self, grad: np.ndarray)
    update_parameters(self,updates: Tuple[np.ndarray])
    zero_grad(self)
    get_derivatives(self)
    get_parameteters(self)
    """

    def __init__(self, n_channels : int, 
                 out_channels : int, 
                 kernel_size : Union[int,List,Tuple], 
                 stride : Union[int,List,Tuple]=1, 
                 padding : Union[int,List,Tuple]=0,
                 padding_value : float=0, 
                 dilation : Union[int,List,Tuple]=1, 
                 output_padding : Union[int,List,Tuple]=0) -> None:

        """Initialize necesary parameters for the Layer
        Paramters:
            n_channels: int
                number of input channels
            out_channels :int
                number of output channels
            kernel_size: int/list/tuple
                kernel size
            dilation : int/list/tuple
                filter dilation
            padding : int/list/tuple
                padding size
            padding_value: float
                padding value
        
        """
        self.n_channels = n_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.padding_value = padding_value
        self.stride = stride
        self.output_padding = output_padding

        if isinstance(padding, int): # if integer is provided as input make it a tuple
            self.padding = tuple([padding]*2)

        if isinstance(kernel_size, int): # if integer is provided as input make it a tuple
            self.kernel_size = tuple([kernel_size]*2)

        if isinstance(stride, int): # if integer is provided as input make it a tuple
            self.stride = tuple([stride]*2)

        if isinstance(output_padding, int): # if integer is provided as input make it a tuple
            self.output_padding= tuple([output_padding]*2)
        
        #initialize parameters using xavier normal
        k = 1/(self.out_channels *  np.sum(self.kernel_size))
        self.w = np.random.uniform(-np.sqrt(k), np.sqrt(k), size=(n_channels, out_channels, self.kernel_size[0], self.kernel_size[1]))
        self.b = np.random.uniform(-np.sqrt(k), np.sqrt(k), size=(out_channels))

        #initialize parameters to 0
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)


    def __parameters_gradient(self, grad):
        """
        Perform gradient of parameters of parameters
        """
        _,_,hin,win = grad.shape
        _,_,hout, wout = self.__dilate_filter().shape
        self.db = grad.sum(axis=(2,3)).mean(axis=0)

        dilated_filter = self.__dilate_filter()
        dw_dilated = np.zeros_like(dilated_filter)
        for h in range(hout):
            h_start = h
            h_end = h_start + hin
            for w in range(wout):
                w_start = w
                w_end = w_start + win
                window = self.x[:,:,h_start:h_end, w_start:w_end]
                tmp = np.tensordot(np.mean(window,axis=0), np.mean(grad,axis=0), axes=((2,1),(2,1)))
                dw_dilated[:,:,h,w] = tmp 
        self.dw = dw_dilated[:,:,::self.stride[0],::self.stride[1]]

    def __calculate_dx(self,grad):
        """
        Calculate derivative with respect to the input
        """
        dilated_filter = self.__dilate_filter()
        kernel_size = tuple([dilated_filter.shape[-1]]*2)
        b,c,hin,win = grad.shape
        newhin = hin + 2 * (kernel_size[0] - 1)
        newwin = win + 2 * (kernel_size[1] - 1)
        new_grad = np.zeros((b,c,newhin,newwin))
        new_grad[:,:,kernel_size[0] - 1:-kernel_size[0] + 1, kernel_size[1] - 1:-kernel_size[1] + 1] = grad
        _,_,hout, wout = self.x.shape
        dx = np.zeros_like(self.x)

        for h in range(hout):
            h_start = h
            h_end = h_start + kernel_size[0]
            for w in range(wout):
                w_start = w
                w_end = w_start + kernel_size[1]
                window = new_grad[:,:,h_start:h_end, w_start:w_end]
                tmp = np.tensordot(window, dilated_filter, axes=((3,2,1),(3,2,1)))
                dx[:,:,h,w] = tmp
        if self.new_padding[0] or self.new_padding[1]:
            return dx[:,:, self.new_padding[0]:-self.new_padding[0]:self.stride[0], self.new_padding[1]:-self.new_padding[1]:self.stride[1]]
        return dx

    
    def __convolution(self, x, weight,bias, hout, wout, c_out, kernel_size,stride, axes=((3,2,1),(3,2,0))):
        """
        Perform convolution using tensordot
        """
        b,_,_,_ =  x.shape
        out = np.zeros((b,c_out, hout, wout))
        for h in range(hout):
            h_start = h * stride[0]
            h_end = h_start + kernel_size[0]
            for w in range(wout):
                w_start = w * stride[1]
                w_end = w_start + kernel_size[1]
                window = x[:,:,h_start:h_end, w_start:w_end]
                tmp = np.tensordot(window, weight, axes=axes) + bias
                out[:,:,h,w] += tmp

        return out
    
    def __dilate_filter(self):
        cout, cin, k, k = self.w.shape
        if self.dilation > 1:
            newk = k + (self.dilation - 1) * (k - 1)
            dilated_filter = np.zeros((cout, cin, newk, newk))
            dilated_filter[:,:,::self.dilation,::self.dilation] = self.w
        else:
            dilated_filter = self.w
        return dilated_filter
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Take input x(size of B,Cin,Hin,Win) and return the forward pass(size B,Cout, Hout, Wout)
        Parameters
        ----------
        x : numpy array
            input to be convolved

        Returns
        -------
        Numpy array: results of convolution
        """
       
        b,cin, hin, win =  x.shape
        hout = (hin - 1) * self.stride[0] - 2 * self.padding[0] + self.dilation * (self.kernel_size[0] - 1)  + self.output_padding[0] + 1
        wout = (win - 1) * self.stride[1] - 2 * self.padding[1] + self.dilation * (self.kernel_size[1] - 1)  + self.output_padding[0] + 1

        dilated_filter = self.__dilate_filter()
        kernel_size = dilated_filter.shape[-2:]
        dilation = tuple([s - 1 for s  in self.stride]) # calculate dilation size
        self.new_padding = tuple([(k -p - 1)  for k,p in zip(kernel_size, self.padding)])  # caluculate new padding
        self.x = np.zeros((b,cin, hin + 2 * self.new_padding[0] + self.output_padding[0] + (hin - 1) * (self.stride[0] - 1), win + self.output_padding[1] + (win - 1) * (self.stride[0] - 1) + 2 * self.new_padding[1]))

        self.x[:,:, self.new_padding[0]:-self.new_padding[0]:dilation[0] + 1, self.new_padding[1]:-self.new_padding[1]:dilation[1] + 1] = x
        stride = (1,1)
        # rotate filter by 180
        flipped_filter = np.rot90(dilated_filter,k=2, axes=(2,3))
        out = self.__convolution(self.x, flipped_filter,self.b, hout, wout, self.out_channels, kernel_size,stride)        
        return out
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)
    

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Take input grad(size of B,Cout,Hout,Wout) and return the packward pass(size B,Cin, Hin, Win)
        Parameters
        ----------
        grad : numpy array
            gradient of the preceding layer

        Returns
        -------
            Numpy array: gradient to be propagated to the previous layer
        """
        # calculate derivative with respect to filter: conv(input,grad)
        self.__parameters_gradient(grad)
        # calculate derivative with respect to input: full conv(rotated 180 filter, grad)
        return self.__calculate_dx(grad)

    def update_parameters(self,updates: Tuple[np.ndarray] ) -> None:
        "Update parameters, Recieves updates"

        self.w -= updates[0]
        self.b -= updates[1]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        "Enable object to be called as a method"
        return self.forward(x)

    def zero_grad(self) -> None:
        "Reset gradient"
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)

    def get_derivatives(self):
        "Return derivatives"
        return self.dw, self.db
    
    def get_parameteters(self):
        "Return parameters"
        return self.w, self.b
