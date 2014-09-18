'''

@author: "epsilonyuan@gmail.com"
'''
from numpy.polynomial import Polynomial
import numpy as np
from pymfr.misc.mixin import SPMixin

class TripleSpline:
    def __init__(self):
        self._left = Polynomial([2 / 3, 0, -4, 4])
        self._right = Polynomial([4 / 3, -4, 4, -4 / 3])
    
    def __call__(self, x):
        if x < 0:
            raise ValueError()
        if x <= 0.5:
            return self._left(x)
        elif x < 1:
            return self._right(x)
        else:
            return 0

    @classmethod
    def deriv(cls):
        res = TripleSpline()
        res._left = res._left.deriv()
        res._right = res._right.deriv()
        return res

def triple_spline(x):
    if x >= 1:
        return 0
    elif x > 0.5:
        return 4 / 3 - 4 * x + 4 * x ** 2 - 4 / 3 * x ** 3
    elif x >= 0:
        return 2 / 3 - 4 * x ** 2 + 4 * x ** 3
    else:
        raise ValueError()

def triple_spline_diff(x):
    if x >= 1:
        return 0
    elif x > 0.5:
        return -4 + 8 * x - 4 * x ** 2
    elif x >= 0:
        return -8 * x + 12 * x ** 2
    else:
        raise ValueError()

triple_spline.deriv = lambda :triple_spline_diff
        

class WeightFunction(SPMixin):
    def __init__(self, node_regular_dist_func, core=None, spatial_dim=2, partial_order=1):
        if not core:
            core = triple_spline
        self.core = core
        self.core_deriv = core.deriv()
        self.node_regular_dist_func = node_regular_dist_func
        
        SPMixin.__init__(self, spatial_dim, partial_order)
    
    
    
    @SPMixin.spatial_dim.setter
    def spatial_dim(self, value):
        SPMixin.spatial_dim.fset(self, value)
        self.node_regular_dist_func.spatial_dim = value
    
    @SPMixin.partial_order.setter
    def partial_order(self, value):
        SPMixin.partial_order.fset(self, value)
        self.node_regular_dist_func.partial_value = value
    
    def __call__(self, x, index, out=None):
        partial_size = self.partial_size()
        r_dists = self.node_regular_dist_func(x, index)
        
        if not out:
            out = np.empty((partial_size,))
        
        r = r_dists[0]
        out[0] = self.core(r)
        td = self.core_deriv(r)
        for j in range(1, partial_size):
            out[j] = td * r_dists[j]
        return out

class RegularNodeRadiusBasedDistanceFunction(SPMixin):
    
    def __init__(self, coord_rad_getter, spatial_dim=2, partial_order=1):
        self.coord_rad_getter = coord_rad_getter
        SPMixin.__init__(self, spatial_dim, partial_order)
    
    def _dist(self, diff):
        spatial_dim = self.spatial_dim
        if spatial_dim == 1:
            return abs(diff[0])
        else:
            return np.dot(diff, diff) ** 0.5
    
    def __call__(self, x, index, out=None):
        coord, rad = self.coord_rad_getter(index)
        
        diff = x - coord
        dist = self._dist(diff)
        
        if out is None:
            out = np.empty((self.partial_size(),))
        
        if self.partial_order >= 1:
            if 0 == dist:
                out.fill(0)
            else:
                out[1:] = diff
                np.divide(out, rad * dist, out)
        out[0] = dist / rad;
        return out

class AllCoordRadiusGetter:
    def __init__(self, all_coord_radius):
        self.all_coord_radius = all_coord_radius
        
    def coord(self, index):
        return self.all_coord_radius[index][0]
    
    def radius(self, index):
        return self.all_coord_radius[index][1]
    
    def all_coord_radius(self, index):
        return self.self(index)
    
    def __call__(self, index):
        return self.all_coord_radius[index]
    
def weight_function(all_coord_radius, core=None):
    return WeightFunction(
              RegularNodeRadiusBasedDistanceFunction(
                 AllCoordRadiusGetter(all_coord_radius)),
              core)
    
