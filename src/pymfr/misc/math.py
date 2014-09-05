'''
@author: "epsilonyuan@gmail.com"
'''

from scipy.misc import comb

def partial_size(spatial_dim, partial_order):
    result = 0
    for i in range(partial_order + 1):
        result += comb(i + spatial_dim - 1, spatial_dim - 1, True)
    return result
