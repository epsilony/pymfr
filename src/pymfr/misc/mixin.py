'''

@author: "epsilonyuan@gmail.com"
'''
from pymfr.misc.math import partial_size
import itertools
import inspect

class SPMixin:
    
    def __init__(self, spatial_dim=2, partial_order=1):
        self.partial_order = partial_order
        self.spatial_dim = spatial_dim
    
    @property
    def partial_order(self):
        return self._partial_order
    
    @partial_order.setter
    def partial_order(self, value):
        if value != 0 and value != 1:
            raise ValueError()
        self._partial_order = value
    
    @property
    def spatial_dim(self):
        return self._spatial_dim
    
    @spatial_dim.setter
    def spatial_dim(self, value):
        if value < 0 or not isinstance(value, int):
            raise ValueError()
        self._spatial_dim = value     
           
    def partial_size(self):
        return partial_size(self.spatial_dim, self.partial_order)

class SetupMixin:
    
    # __prerequisites__ = []
    # __optionals__ = []
    # __after_setup__= []
    
    
    def setup(self, **kwargs):
        
        for name in getattr(self, '__prerequisites__', ()):
            setattr(self, name, kwargs[name])
        
        for name, default in getattr(self, '__optionals__', ()):
            if name not in kwargs or kwargs[name] is None:
                if not hasattr(self, name):
                    setattr(self, name, default)
            else:
                setattr(self, name, kwargs[name])
                
        
        for f in getattr(self, '__after_setup__', ()):
            if isinstance(f, str):
                f_name = f
                f = getattr(self, f_name, None)
            if inspect.isfunction(f):
                f(self)
            elif inspect.ismethod(f):
                f()
        
        return self
    
    def clear(self):
        for name in itertools.chain[self.__prerequisites__, self.__optionals__]:
            if hasattr(self, name):
                delattr(self, name)
            
            
         
