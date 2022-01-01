import torch
import numpy as np
from .base import BaseLinOp


class Matrix(BaseLinOp):
    def __init__(self, matrix):
        self.H = matrix
        self.in_size = matrix.shape[1]
        self.out_size = matrix.shape[0]
    
    def apply(self, x):
        return self.H @ x

    def applyAdjoint(self, x):
        return self.H.T.conj() @ x

class Multiplication(BaseLinOp):
    """coefs is a vector for element-wise multiplication"""
    def __init__(self, coefs):
        self.coefs = coefs
        self.in_size = coefs.shape[0]
        self.out_size = coefs.shape[0]

    def apply(self, x):
        return self.coefs * x

    def applyAdjoint(self, x):
        return self.coefs.conj() * x
    
class Conv(BaseLinOp):
    """Convolution operator, computed using FFT"""
    def __init__(self, Ffilter, epsilon=1e-5):
        self.Ffilter = Ffilter  # Filter in the Fourier domain
        self.epsilon = epsilon  # regularization constant for deconvolution
        self.in_size = Ffilter.shape[0]
        self.out_size = Ffilter.shape[0]

    def apply(self, x):
        return torch.fft.ifft2(self.Ffilter * torch.fft.fft2(x))

    def applyAdjoint(self, x):
        return torch.fft.fft2(1 / (self.Ffilter+self.epsilon) * torch.fft.ifft2(x))

class FFT(BaseLinOp):
    def __init__(self):
        self.in_size = -1
        self.out_size = -1

    def apply(self, x):
        return torch.fft.fft(x, norm="ortho")

    def applyAdjoint(self, x):
        return torch.fft.ifft(x, norm="ortho")

class IFFT(BaseLinOp):
    def __init__(self):
        self.in_size = -1
        self.out_size = -1

    def apply(self, x):
        return torch.fft.ifft(x, norm="ortho")

    def applyAdjoint(self, x):
        return torch.fft.fft(x, norm="ortho")

class FFT2(BaseLinOp):
    def __init__(self):
        self.in_size = -1
        self.out_size = -1

    def apply(self, x):
        return torch.fft.fft2(x, norm="ortho")

    def applyAdjoint(self, x):
        return torch.fft.ifft2(x, norm="ortho")

class IFFT2(BaseLinOp):
    def __init__(self):
        self.in_size = -1
        self.out_size = -1

    def apply(self, x):
        return torch.fft.ifft2(x, norm="ortho")

    def applyAdjoint(self, x):
        return torch.fft.fft2(x, norm="ortho")
    
class Id(BaseLinOp):
    def __init__(self):
        self.in_size = -1
        self.out_size = -1
        
    def apply(self, x):
        return x
    
    def applyAdjoint(self, x):
        return x
    
class Constant(BaseLinOp):
    """WARNING: THIS IS NOT A LINOP, TO BE DISCUSSED"""
    def __init__(self, value=1):
        self.value = value
        self.in_size = -1
        self.out_size = -1
        
    def apply(self, x):
        return self.value
    
    def applyAdjoint(self, x):
        return 0
    
class Flip(BaseLinOp):
    def __init__(self):
        self.in_size = -1
        self.out_size = -1

    def apply(self, x):
        return torch.flip(x, dims=[i for i in range(x.dim())])

    def applyAdjoint(self, x):
        return torch.flip(x, dims=[i for i in range(x.dim())])

class Roll(BaseLinOp):
    def __init__(self, shifts, dims):
        self.in_size = -1
        self.out_size = -1
        self.shifts = shifts
        self.dims = dims

    def apply(self, x):
        return torch.roll(x, shifts=self.shifts, dims=self.dims)

    def applyAdjoint(self, x):
        return torch.roll(x, shifts=[-shift for shift in self.shifts], dims=self.dims)
    
class StackLinOp(BaseLinOp):
    def __init__(self, LinOp1, LinOp2):
        self.LinOp1 = LinOp1
        self.LinOp2 = LinOp2
        self.in_size = np.maximum(LinOp1.in_size, LinOp2.in_size)
        self.out_size = LinOp1.out_size + LinOp2.out_size  # TODO: handle cases where out_size is undefined

    def apply(self, x):
        return torch.cat((self.LinOp1.apply(x), self.LinOp2.apply(x)), dim=0)

    def applyAdjoint(self, x):
        return self.LinOp2.applyAdjoint(x[:self.LinOp2.out_size]) + self.LinOp1.applyAdjoint(x[self.LinOp2.out_size:])
    
class Adjoint(BaseLinOp):
    def __init__(self, LinOp):
        self.LinOp = LinOp
        self.in_size = LinOp.out_size
        self.out_size = LinOp.in_size

    def apply(self, x):
        return self.LinOp.applyAdjoint(x)

    def applyAdjoint(self, x):
        return self.LinOp.apply(x)

