'''

@author: "epsilonyuan@gmail.com"
'''
from abc import ABCMeta, abstractmethod
from pymfr.misc.math import partial_size
import numpy as np
from pymfr.misc.mixin import SetupMixin

class ShapeFunctionCore(metaclass=ABCMeta):
    
    @abstractmethod
    def calc(self, x, node_coords, node_indes):
        pass
    
class ShapeFunction(metaclass=ABCMeta):
    
    @abstractmethod
    def __call__(self, node_indes):
        pass

class CoredShapeFunction(ShapeFunction, SetupMixin):
    __prerequisites__ = ['node_coords', 'spatial_dim']
    __optionals__ = [('partial_order', 0)]
    
    def __init__(self, shape_function_core):
        self.shape_function_core = shape_function_core
        
    @property
    def partial_order(self):
        return self.shape_function_core.partial_order
    
    @partial_order.setter
    def partial_order(self, value):
        self.shape_function_core.parital_order = value
    
    @property
    def spatial_dim(self):
        return self.shape_function_core.spatial_dim
    
    @spatial_dim.setter
    def spatial_dim(self, value):
        self.shape_function_core.spatial_dim = value
    
    def __call__(self, x, node_indes):
        return self.shape_function_core.calc(x, self.node_coords[node_indes], node_indes)

class LinearShapeFuntion(ShapeFunction, SetupMixin):
    __prerequisites__ = ['node_coords', 'spatial_dim']
    __optionals__ = [('partial_order', 0)]
    
    def __init__(self, spatial_dim=2, partial_order=0):
        self.spatial_dim = spatial_dim
        self.partial_order = partial_order
    
    def __call__(self, x, node_indes):
        if len(node_indes) != 2:
            raise ValueError()
        
        size = partial_size(self.spatial_dim, self.partial_order)
        ret = np.empty((size, 2))
        
        coord_0, coord_1 = self.node_coords[node_indes]
        
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
    
class SoloShapeFunction(ShapeFunction, SetupMixin):
    __prerequisites__ = ['spatial_dim']
    __optionals__ = [('partial_order', 0)]    
    def __call__(self, x, node_indes):
        if len(node_indes) != 1:
            raise ValueError()
        
        size = partial_size(self.spatial_dim, self.partial_order)
        ret = np.zeros((size, 1))
        ret[0, 0] = 1
        
        return ret

class RawCoordRadiusGetter(SetupMixin):
    
    __prerequisites__ = ['node_coords', 'node_radiuses']
    
    def __call__(self, index):
        return (self.node_coords[index], self.node_radiuses[index])
    
    
