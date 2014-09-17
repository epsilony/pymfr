'''

@author: "epsilonyuan@gmail.com"
'''
from abc import ABCMeta, abstractmethod
from pymfr.misc.mixin import SetupMixin

class LoadCalculator(metaclass=ABCMeta):
    
    @abstractmethod
    def __call__(self, load_key, coord, bnd):
        pass

class SimpLoadCalculator(LoadCalculator, SetupMixin):
    
    __prerequisites__ = ['load_map']
    
    def __init__(self, load_map=None):
        if load_map is None:
            self.load_map = {}
        else:
            self.load_map = load_map
    
    def __call__(self, load_key, coord, bnd):
        load_core = self.load_map.get(load_key, None)
        if load_core is None:
            return None
        else:
            return load_core(coord, bnd)
