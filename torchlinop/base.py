class BaseLinOp:
    def __init__(self):
        pass

    @property
    def T(self):
        return Adjoint(self)

    def __add__(self, other):
        if isinstance(other, BaseLinOp):
            return Sum(self, other)
        else:
            return ScalarSum(self, other)

    def __radd__(self, other):
        if isinstance(other, BaseLinOp):
            return Sum(self, other)
        else:
            return ScalarSum(self, other)
    
    def __sub__(self, other):
        if isinstance(other, BaseLinOp):
            return Diff(self, other)
        else:
            return ScalarDiff(self, other)
    
    def __rsub__(self, other):
        if isinstance(other, BaseLinOp):
            return Diff(self, other)
        else:
            return ScalarDiff(self, other)
        
    def __mul__(self, other):
        if isinstance(other, BaseLinOp):
            raise NameError('The multiplication operator can only be performed between a LinOp object and a scalar or vector.')
        else:
            return ScalarMul(self, other)
        
    def __rmul__(self, other):
        if isinstance(other, BaseLinOp):
            raise NameError('The multiplication operator can only be performed between a LinOp object and a scalar or vector.')
        else:
            return ScalarMul(self, other)
    
    def __matmul__(self, other):
        if isinstance(other, BaseLinOp):
            return Composition(self, other)
        else:
            raise NameError('The matrix multiplication operator can only be performed between two LinOp objects.')


class Composition(BaseLinOp):
    def __init__(self, LinOp1, LinOp2):
        if LinOp2.out_size != LinOp1.in_size and LinOp2.out_size != -1 and LinOp1.in_size != -1:
            raise NameError('The dimensions of the LinOp composition do not match.')
        self.LinOp1 = LinOp1
        self.LinOp2 = LinOp2
        self.in_size = LinOp2.in_size if LinOp2.in_size != -1 else LinOp1.in_size
        self.out_size = LinOp1.out_size if LinOp1.out_size != -1 else LinOp2.out_size

    def apply(self, x):
        return self.LinOp1.apply(self.LinOp2.apply(x))

    def applyAdjoint(self, x):
        return self.LinOp2.applyAdjoint(self.LinOp1.applyAdjoint(x))

class Sum(BaseLinOp):
    def __init__(self, LinOp1, LinOp2):
        if LinOp2.in_size != LinOp1.in_size and LinOp2.in_size != -1 and LinOp1.in_size != -1:
            raise NameError('The input dimensions of the LinOp sum do not match.')
        if LinOp2.out_size != LinOp1.out_size and LinOp2.out_size != -1 and LinOp1.out_size != -1:
            raise NameError('The output dimensions of the LinOp sum do not match.')
        self.LinOp1 = LinOp1
        self.LinOp2 = LinOp2
        self.in_size = max(LinOp1.in_size, LinOp2.in_size)  # it is -1 if size is undefined
        self.out_size = max(LinOp1.out_size, LinOp2.out_size)

    def apply(self, x):
        return self.LinOp1.apply(x) + self.LinOp2.apply(x)

    def applyAdjoint(self, x):
        return self.LinOp2.applyAdjoint(x) + self.LinOp1.applyAdjoint(x)
    
class ScalarSum(BaseLinOp):
    def __init__(self, LinOp, other):
        self.LinOp = LinOp
        self.other = other
        self.in_size = LinOp.in_size
        self.out_size = LinOp.out_size

    def apply(self, x):
        return self.LinOp.apply(x) + self.other

    def applyAdjoint(self, x):
        return self.LinOp.applyAdjoint(x)

class Diff(BaseLinOp):
    def __init__(self, LinOp1, LinOp2):
        if LinOp2.in_size != LinOp1.in_size and LinOp2.in_size != -1 and LinOp1.in_size != -1:
            raise NameError('The input dimensions of the LinOp sum do not match.')
        if LinOp2.out_size != LinOp1.out_size and LinOp2.out_size != -1 and LinOp1.out_size != -1:
            raise NameError('The output dimensions of the LinOp sum do not match.')
        self.LinOp1 = LinOp1
        self.LinOp2 = LinOp2
        self.in_size = max(LinOp1.in_size, LinOp2.in_size)
        self.out_size = max(LinOp1.out_size, LinOp2.out_size)

    def apply(self, x):
        return self.LinOp1.apply(x) - self.LinOp2.apply(x)

    def applyAdjoint(self, x):
        return self.LinOp1.applyAdjoint(x) - self.LinOp2.applyAdjoint(x)

class ScalarDiff(BaseLinOp):
    def __init__(self, LinOp, other):
        self.LinOp = LinOp
        self.other = other
        self.in_size = LinOp.in_size
        self.out_size = LinOp.out_size

    def apply(self, x):
        return self.LinOp.apply(x) - self.other

    def applyAdjoint(self, x):
        return self.LinOp.applyAdjoint(x)

class ScalarMul(BaseLinOp):
    def __init__(self, LinOp, other):
        self.LinOp = LinOp
        self.other = other
        self.in_size = LinOp.in_size
        self.out_size = LinOp.out_size

    def apply(self, x):
        return self.other * self.LinOp.apply(x) 

    def applyAdjoint(self, x):
        return self.other * self.LinOp.applyAdjoint(x)

class Adjoint(BaseLinOp):
    def __init__(self, LinOp):
        self.LinOp = LinOp
        self.in_size = LinOp.out_size
        self.out_size = LinOp.in_size

    def apply(self, x):
        return self.LinOp.applyAdjoint(x)

    def applyAdjoint(self, x):
        return self.LinOp.apply(x)

