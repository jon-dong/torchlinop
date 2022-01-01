from .utils import *


class BaseLinOp:
    def __init__(self):
        pass

    def __add__(self, other):
        if isinstance(other, BaseLinOp):
            return LinOpSum(self, other)
        else:
            return LinOpScalarSum(self, other)

    def __radd__(self, other):
        if isinstance(other, BaseLinOp):
            return LinOpSum(self, other)
        else:
            return LinOpScalarSum(self, other)
    
    def __sub__(self, other):
        if isinstance(other, BaseLinOp):
            return LinOpDiff(self, other)
        else:
            return LinOpScalarDiff(self, other)
    
    def __rsub__(self, other):
        if isinstance(other, BaseLinOp):
            return LinOpDiff(self, other)
        else:
            return LinOpScalarDiff(self, other)
        
    def __mul__(self, other):
        if isinstance(other, BaseLinOp):
            raise NameError('This case of multiplication between two LinOp objects has not been implemented yet. Please have a look at LinOpMul and contact us if you need additional functionalities.')
        else:
            return LinOpScalarMul(self, other)
        
    def __rmul__(self, other):
        if isinstance(other, BaseLinOp):
            raise NameError('This case of multiplication between two LinOp objects has not been implemented yet. Please have a look at LinOpMul and contact us if you need additional functionalities.')
        else:
            return LinOpScalarMul(self, other)
    
    def __matmul__(self, other):
        if isinstance(other, BaseLinOp):
            return LinOpComposition(self, other)
        else:
            raise NameError('The matrix multiplication operator can only be performed between two LinOp objects.')
