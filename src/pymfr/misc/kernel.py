'''

@author: "epsilonyuan@gmail.com"
'''
from pymfr.misc.mixin import SPMixin
from pymfr.misc.math import partial_size
import numpy as np
from pymfr.misc.tools import FieldProxy

class PolynomialKernel:
    
    def __init__(self, order=1, spatial_dim=2, partial_order=1):
        self._kernels = (PolynomialKernel1D(), PolynomialKernel2D())
        self._kernel = self._kernels[spatial_dim - 1]
        self._kernel.order = order
        self._kernel.partial_order = partial_order
        self.spatial_dim = spatial_dim
        self._is_linear_transformable = True
        
    @property
    def is_linear_transformable(self):
        return self._is_linear_transformable
    
    @property
    def spatial_dim(self):
        return self._spatial_dim
    
    @spatial_dim.setter
    def spatial_dim(self, value):
        self._spatial_dim = value
        old_order = self._kernel.order
        old_partial_order = self._kernel.partial_order
        
        self._kernel = self._kernels[value - 1]
        
        self._kernel.partial_order = old_partial_order
        self._kernel.order = old_order
        
    def __call__(self, x):
        return self._kernel(x)

for proxied in ('order', 'partial_order', 'size'):
    setattr(PolynomialKernel, proxied, FieldProxy('_kernel', proxied))

class _PolynomialKernelBase:
    
    def __init__(self, spatial_dim, order=1, partial_order=1):
        self.order = order
        self._max_order = 3
        self._max_partial_order = 1
        self.partial_order = partial_order
        self._spatial_dim = spatial_dim
        
    @property
    def order(self):
        return self._order
    
    @property
    def max_partial_order(self):
        return self._max_partial_order
    
    @property
    def max_order(self):
        return self._order
    
    @order.setter
    def order(self, value):
        self._order = value
    
    @property
    def partial_order(self):
        return self._partial_order
    
    @partial_order.setter
    def partial_order(self, value):
        if not isinstance(value, int) or value < 0 or value > 1:
            raise ValueError()
        self._partial_order = value
    
    @property
    def spatial_dim(self):
        return self._spatial_dim
    
    def partial_size(self):
        return partial_size(self.spatial_dim, self.partial_order)
    
    @property
    def size(self):
        return partial_size(self.spatial_dim, self.order)
    
class PolynomialKernel1D(_PolynomialKernelBase):
    
    def __init__(self, order=1, partial_order=1):
        super(PolynomialKernel1D, self).__init__(1, order, partial_order)
        self._calcs = (
             self.__calc10__,
             self.__calc11__,
             self.__calc20__,
             self.__calc21__,
             self.__calc30__,
             self.__calc31__)
        
    def __call__(self, x, out=None):
        if not out:
            out = np.empty((self.partial_size(), self.size))
        magic = 2
        calc = self._calcs[(self.order - 1) * magic + self.partial_order]
        calc(x, out)
        return out
    
    def __calc10__(self, x, out):
        out[0, 0], out[0, 1] = 1, x
    
    def __calc11__(self, x, out):
        self.__calc10__(x, out)
        out[1, 0], out[1, 1] = 0, 1
    
    def __calc20__(self, x, out):
        out[0, 0], out[0, 1], out[0, 2] = 1, x, x * x
    
    def __calc21__(self, x, out):
        self.__calc20__(x, out)
        out[1, 0], out[1, 1], out[1, 2] = 0, 1, 2 * x
    
    def __calc30__(self, x, out):
        out[0, 0], out[0, 1], out[0, 2], out[0, 3] = 1, x, x * x, x * x * x
        
    def __calc31__(self, x, out):
        self.__calc30__(x, out)
        out[1, 0], out[1, 1], out[1, 2], out[1, 3] = 0, 1, 2 * x, 3 * x * x
        
class PolynomialKernel2D(_PolynomialKernelBase):
    
    def __init__(self, order=1, partial_order=1):
        super(PolynomialKernel2D, self).__init__(2, order, partial_order)
        self._calcs = (
             self.__calc10__,
             self.__calc11__,
             self.__calc20__,
             self.__calc21__,
             self.__calc30__,
             self.__calc31__)
        
    def __call__(self, x, out=None):
        if not out:
            out = np.empty((self.partial_size(), self.size))
        magic = 2
        calc = self._calcs[(self.order - 1) * magic + self.partial_order]
        calc(x, out)
        return out

    def __calc10__(self, x, out):
        out[0, 0], out[0, 1], out[0, 2] = 1, x[0], x[1]
        
    
    def __calc11__(self, x, out):
        out[0, 0], out[0, 1], out[0, 2] = 1, x[0], x[1]
        out[1, 0], out[1, 1], out[1, 2] = 0, 1, 0
        out[2, 0], out[2, 1], out[2, 2] = 0, 0, 1
    
    def __calc20__(self, x, out):
        x0 = x[0]
        x1 = x[1]
        out[0, 0], out[0, 1], out[0, 2], out[0, 3], out[0, 4], out[0, 5] = 1, x0, x1, x0 * x0, x0 * x1, x1 * x1
        
    def __calc21__(self, x, out):
        x0 = x[0]
        x1 = x[1]
        out[0, 0], out[0, 1], out[0, 2], out[0, 3], out[0, 4], out[0, 5] = 1, x0, x1, x0 * x0, x0 * x1, x1 * x1
        out[1, 0], out[1, 1], out[1, 2], out[1, 3], out[1, 4], out[1, 5] = 0, 1, 0, 2 * x0, x1, 0
        out[2, 0], out[2, 1], out[2, 2], out[2, 3], out[2, 4], out[2, 5] = 0, 0, 1, 0, x0, 2 * x1   
    
    def __calc30__(self, x, out):
        self.__calc20__(x, out)
        out[0][6:] = (x[0] ** 3, x[0] ** 2 * x[1], x[0] * x[1] ** 2, x[1] ** 3)
    
    def __calc31__(self, x, out):
        x0 = x[0]
        x1 = x[1]
        out[0, 0], out[0, 1], out[0, 2], out[0, 3], out[0, 4], out[0, 5] = 1, x0, x1, x0 * x0, x0 * x1, x1 * x1
        out[1, 0], out[1, 1], out[1, 2], out[1, 3], out[1, 4], out[1, 5] = 0, 1, 0, 2 * x0, x1, 0
        out[2, 0], out[2, 1], out[2, 2], out[2, 3], out[2, 4], out[2, 5] = 0, 0, 1, 0, x0, 2 * x1   
        
        out[0, 6], out[0, 7], out[0, 8], out[0, 9] = x0 * x0 * x0, x0 * x0 * x1, x0 * x1 * x1, x1 * x1 * x1
        out[1, 6], out[1, 7], out[1, 8], out[1, 9] = 3 * x0 * x0, 2 * x0 * x1, x1 * x1, 0
        out[2, 6], out[2, 7], out[2, 8], out[2, 9] = 0, x0 * x0, 2 * x0 * x1, 3 * x1 * x1
        
    
        
