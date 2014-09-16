'''

@author: "epsilonyuan@gmail.com"
'''
from abc import ABCMeta, abstractmethod
from pymfr.misc.math import partial_size
import numpy as np
from pymfr.misc.mixin import SetupMixin

class ShapeFunctionCore(metaclass=ABCMeta):
    
    @abstractmethod
    def calc(self, x, coords, node_indes):
        pass
    
class ShapeFunction(metaclass=ABCMeta):
    
    @abstractmethod
    def __call__(self, node_indes):
        pass

class CoredShapeFunction(ShapeFunction, SetupMixin):
    __prerequisites__ = ['coords', 'spatial_dim']
    __optionals__ = [('partial_order', 0)]
        
    @property
    def partial_order(self):
        return self.shape_func_core.partial_order
    
    @partial_order.setter
    def partial_order(self, value):
        self.shape_func_core.parital_order = value
    
    @property
    def spatial_dim(self):
        return self.shape_func_core.spatial_dim
    
    @spatial_dim.setter
    def spatial_dim(self, value):
        self.shape_func_core.spatial_dim = value
    
    def __call__(self, x, node_indes):
        return self.shape_func_core.calc(x, self.coords[node_indes], node_indes)

class LinearShapeFuntion(ShapeFunction):
    
    def __init__(self, spatial_dim=1, partial_order=0):
        self.spatial_dim = spatial_dim
        self.partial_order = partial_order
    
    def setup(self, **kwargs):
        self.coords = kwargs['coords']
        self.spatial_dim = kwargs.get('spatial_dim', 1)
        self.partial_order = kwargs.get('partial_order', 0)
    
    def __call__(self, x, node_indes):
        if len(node_indes) != 2:
            raise ValueError()
        
        size = partial_size(self.spatial_dim, self.partial_order)
        ret = np.empty((size, 2))
        
        coord_0, coord_1 = self.coords[node_indes]
        
        length = np.linalg.norm(coord_0 - coord_1)
        det = x - coord_0
        det_len = np.linalg.norm(det)
        t = det_len / length
        
        ret[0, 0] = 1 - t
        ret[0, 1] = t
        
        if self.partial_order > 0:
            ret[1:1 + self.spatial_dim, 0] = -det / det_len / length
            ret[1:1 + self.spatial_dim, 1] = det / det_len / length
        
        return ret
