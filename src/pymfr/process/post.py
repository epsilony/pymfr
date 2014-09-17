'''

@author: "epsilonyuan@gmail.com"
'''

import numpy as np
from pymfr.misc.mixin import SetupMixin
class SimpPostProcessor(SetupMixin):
    
    __prerequisites__ = ['node_values']
    
    def __init__(self, support_node_searcher, shape_func):
        self.support_node_searcher = support_node_searcher
        self.shape_func = shape_func
    
    def process(self, coord, bnd):
        node_indes = self.support_node_searcher.search_indes(coord, bnd)
        shape_func_value = self.shape_func(coord, node_indes)
        
        node_values = self.node_values[node_indes]
        
        return np.dot(shape_func_value, node_values.reshape((-1, 1))).reshape(-1)
        