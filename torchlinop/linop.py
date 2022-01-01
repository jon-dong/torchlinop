import torch
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
    def __init__(self, factor):
        self.factor = factor
        self.in_size = factor.shape[0]
        self.out_size = factor.shape[0]

    def apply(self, x):
        return self.factor * x

    def applyAdjoint(self, x):
        return self.factor.conj() * x
    
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

class Stack(BaseLinOp):
    def __init__(self, LinOpList):
        self.LinOpList = LinOpList
        self.in_size = max([LinOp.in_size for LinOp in LinOpList])
        self.out_size = sum([LinOp.out_size if LinOp.out_size != -1 else self.in_size
            for LinOp in LinOpList])
        for LinOp in LinOpList:
            if LinOp.in_size != self.in_size and LinOp.in_size != -1:
                raise NameError('The input dimensions of the LinOp stack are not consistent.')

    def apply(self, x):
        return torch.cat([LinOp.apply(x) for LinOp in self.LinOpList], dim=0)

    def applyAdjoint(self, x):
        splitted_x = torch.split(x, [LinOp.out_size if LinOp.out_size != -1 else self.in_size
            for LinOp in self.LinOpList])
        return sum([LinOp.applyAdjoint(splitted_x[i]) for (i, LinOp) in enumerate(self.LinOpList)])

