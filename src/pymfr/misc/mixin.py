'''

@author: "epsilonyuan@gmail.com"
'''
from pymfr.misc.math import partial_size

class SPMixin:
    
    def __init__(self,spatial_dim=2,partial_order=1):
        self.partial_order = partial_order
        self.spatial_dim = spatial_dim
    
    @property
    def partial_order(self):
        return self._partial_order
    
    @partial_order.setter
    def partial_order(self,value):
        if value !=0 and value !=1:
            raise ValueError()
        self._partial_order=value
    
    @property
    def spatial_dim(self):
        return self._spatial_dim
    
    @spatial_dim.setter
    def spatial_dim(self,value):
        if value<0 or not isinstance(value,int):
            raise ValueError()
        self._spatial_dim=value     
           
    def partial_size(self):
        return partial_size(self.spatial_dim, self.partial_order)   