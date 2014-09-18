'''
@author: "epsilonyuan@gmail.com"
'''

from scipy.misc import comb
import numpy as np

def _slow_partial_size(spatial_dim, partial_order):
    result = 0
    for i in range(partial_order + 1):
        result += comb(i + spatial_dim - 1, spatial_dim - 1, True)
    return result

_partial_size_table = np.array((1, 2, 3, 4, 1, 3, 6, 10, 1, 4, 10, 20), dtype=int)

def partial_size(spatial_dim, partial_order):
    if partial_order == 0:
        return 1
    if partial_order == 1:
        return spatial_dim + 1
    if spatial_dim > 3 or partial_order > 3:
        return _slow_partial_size(spatial_dim, partial_order)
    return _partial_size_table[4 * (spatial_dim - 1) + partial_order]
