from base import BaseLinOp
import numpy as np


class LinOpComposition(BaseLinOp):
    def __init__(self, LinOp1, LinOp2):
        self.LinOp1 = LinOp1
        self.LinOp2 = LinOp2
        self.in_size = LinOp2.in_size if LinOp2.in_size != -1 else LinOp1.in_size
        self.out_size = LinOp1.out_size if LinOp1.out_size != -1 else LinOp2.out_size

    def apply(self, x):
        return self.LinOp1.apply(self.LinOp2.apply(x))

    def applyAdjoint(self, x):
        return self.LinOp2.applyAdjoint(self.LinOp1.applyAdjoint(x))

class LinOpSum(BaseLinOp):
    def __init__(self, LinOp1, LinOp2):
        self.LinOp1 = LinOp1
        self.LinOp2 = LinOp2
        self.in_size = np.maximum(LinOp1.in_size, LinOp2.in_size)  # it is -1 if size is undefined
        self.out_size = np.maximum(LinOp1.out_size, LinOp2.out_size)

    def apply(self, x):
        return self.LinOp1.apply(x) + self.LinOp2.apply(x)

    def applyAdjoint(self, x):
        return self.LinOp2.applyAdjoint(x) + self.LinOp1.applyAdjoint(x)
    
class LinOpScalarSum(BaseLinOp):
    def __init__(self, LinOp, other):
        self.LinOp = LinOp
        self.scalar = other
        self.in_size = LinOp.in_size
        self.out_size = LinOp.out_size

    def apply(self, x):
        return self.LinOp.apply(x) + self.scalar

    def applyAdjoint(self, x):
        return self.LinOp.applyAdjoint(x)

class LinOpDiff(BaseLinOp):
    def __init__(self, LinOp1, LinOp2):
        self.LinOp1 = LinOp1
        self.LinOp2 = LinOp2
        self.in_size = np.maximum(LinOp1.in_size, LinOp2.in_size)
        self.out_size = np.maximum(LinOp1.out_size, LinOp2.out_size)

    def apply(self, x):
        return self.LinOp1.apply(x) - self.LinOp2.apply(x)

    def applyAdjoint(self, x):
        return self.LinOp1.applyAdjoint(x) - self.LinOp2.applyAdjoint(x)

class LinOpScalarDiff(BaseLinOp):
    def __init__(self, LinOp, other):
        self.LinOp = LinOp
        self.scalar = other
        self.in_size = LinOp.in_size
        self.out_size = LinOp.out_size

    def apply(self, x):
        return self.LinOp.apply(x) - self.scalar

    def applyAdjoint(self, x):
        return self.LinOp.applyAdjoint(x)
    
class LinOpScalarMul(BaseLinOp):
    def __init__(self, LinOp, other):
        self.LinOp = LinOp
        self.scalar = other
        self.in_size = LinOp.in_size
        self.out_size = LinOp.out_size

    def apply(self, x):
        return self.LinOp.apply(x) * self.scalar

    def applyAdjoint(self, x):
        return self.LinOp.applyAdjoint(x) * self.scalar
