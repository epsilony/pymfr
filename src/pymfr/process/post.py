'''

@author: "epsilonyuan@gmail.com"
'''

import numpy as np
from pymfr.misc.mixin import SetupMixin
class SimpPostProcessor(SetupMixin):
    
    __prerequisites__ = ['node_values', 'value_dim']
    
    def __init__(self, support_node_searcher, shape_func):
        self.support_node_searcher = support_node_searcher
        self.shape_func = shape_func
    
    def process(self, coord, bnd):
        node_indes = self.support_node_searcher.search_indes(coord, bnd)
        shape_func_value = self.shape_func(coord, node_indes)
        
        node_indes = np.array(node_indes, dtype=int)
        value_dim = self.value_dim
        result = np.empty((value_dim, shape_func_value.shape[0]))
        for i in range(value_dim):
            node_values = self.node_values[node_indes * value_dim + i]
            result[i] = np.dot(shape_func_value, node_values.reshape((-1, 1))).T
        
        return result
        
    @property
    def partial_order(self):
        return self.shape_func.partial_order
    
    @partial_order.setter
    def partial_order(self, value):
        self.shape_func.partial_order = value
