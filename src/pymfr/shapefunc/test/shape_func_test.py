'''

@author: "epsilonyuan@gmail.com"
'''

import numpy as np
from pymfr.shapefunc.shape_func import LinearShapeFuntion
from nose.tools import assert_almost_equal

def test_linear_shape_func():
    coords = np.array([
                     [-1, -2, -3],
                     [2, 3, 4]])
    
    lsf = LinearShapeFuntion()
    
    
    lsf.setup(coords=coords)
    lsf.spatial_dim = 3
    lsf.partial_order = 1
    
    t = 0.3
    
    x = coords[0] * (1 - t) + coords[1] * t
    
    det = x - coords[0]
    length = np.linalg.norm(coords[1] - coords[0])
    d = length * np.linalg.norm(det)
    act = lsf(x, [1, 0])
    exp = np.array([
                  [t, 1 - t],
                  [det[0] / d, -det[0] / d],
                  [det[1] / d, -det[1] / d],
                  [det[2] / d, -det[2] / d]]
                  )
    assert_almost_equal(0, np.linalg.norm(act - exp))
    
