'''

@author: "epsilonyuan@gmail.com"
'''
from collections import Sequence
import numpy as np

class FieldProxy:
    def __init__(self, field_name, proxy_name):
        self.field_name = field_name
        self.proxy_name = proxy_name
        
    def __get__(self, obj, owner=None):
        field = getattr(obj, self.field_name)
        return getattr(field, self.proxy_name)
    
    def __set__(self, obj, value):
        field = getattr(obj, self.field_name)
        setattr(field, self.proxy_name, value)
        
def ensure_sequence(obj):
    return obj if isinstance(obj,Sequence) else list(obj)

def twod_uniform_coords(xs, ys):
    xvs, yvs = np.meshgrid(xs, ys)
    
    coords = np.empty((xvs.size, 2))
    coords[:, 0] = xvs.flat
    coords[:, 1] = yvs.flat
    
    return coords