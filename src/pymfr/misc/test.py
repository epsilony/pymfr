'''
@author: "epsilonyuan@gmail.com"
'''

from nose.tools import eq_, assert_almost_equal, assert_list_equal
from pymfr.misc.math import partial_size
from pymfr.misc.weight_func import TripleSpline, weight_function
import numpy as np
from pymfr.misc.kernel import PolynomialKernel
import random
from pymfr.misc.tools import trans_coords


def test_partial_size():
    # (dim,order,exp)
    datas = [(1, 0, 1), (1, 1, 2), (1, 2, 3), (1, 3, 4),
             (2, 0, 1), (2, 1, 3), (2, 2, 6), (2, 3, 10),
             (3, 0, 1), (3, 1, 4), (3, 2, 10), (3, 3, 20)
           ]
    for dim, order, exp in datas:
        act = partial_size(dim, order)
        eq_(exp, act)
        
def test_triple_spline():
    ts = TripleSpline()
    ts_deriv = ts.deriv()
    
    assert_almost_equal(ts(0), 2 / 3)
    eq_(ts(1), 0)
    eq_(ts(2), 0)
    assert_almost_equal(ts(0.5), 1 / 6)
    assert_almost_equal(ts(0.25), 23 / 48)
    assert_almost_equal(ts(0.8), 0.010666666666666824)
    
    eq_(ts_deriv(0), 0)
    eq_(ts_deriv(1), 0)
    eq_(ts_deriv(2), 0)
    assert_almost_equal(ts_deriv(0.2), -1.12)
    assert_almost_equal(ts_deriv(0.5), -1)
    assert_almost_equal(ts_deriv(0.50000001), -1)
    assert_almost_equal(ts_deriv(0.99999999), 0)

def test_weight_func():
    all_nodes_coords_1d = [([-1], 2), ([-2], 1), ([3], 3)]
    x_index_exp_iter_1d = [
         ([-1], 1, [0, 0]),
         ([-1.5], 1, [1 / 6, -1]),
         ([3], 2, [2 / 3, 0]),
         ([2.25], 2, [23 / 48, 5 / 12]),
         ([-2], 0, [1 / 6, 1 / 2])]
    all_nodes_coords_3d = [([-1, 0, 2], 2), ([-2, 2, 3], 3), ([3, 11, 2], 1)]
    x_index_exp_iter_3d = [
         ([3, 11, 2], 2, [2 / 3, 0, 0, 0]),
         ([-2, 11, 2], 2, [0, 0, 0, 0]),
         ([4, 11, 2], 2, [0, 0, 0, 0]),
         ([3, 11, 1], 2, [0, 0, 0, 0]),
         ([3, 11, 3], 2, [0, 0, 0, 0]),
         ([-1, 1, 2], 1, [0.100665470268330, -0.137511589670446, 0.137511589670446, 0.137511589670446]),
         ([-0.5, 1, 1.5], 1, [0.0138638964379885, -0.0406268387361561, 0.0270845591574374, 0.0406268387361561])
         ]
    for nc, sd, xie in [
              (all_nodes_coords_1d, 1, x_index_exp_iter_1d),
              (all_nodes_coords_3d, 3, x_index_exp_iter_3d)
              ]:
        yield _test_weight_func, nc, sd, xie
    
def _test_weight_func(all_nodes_coords, spatial_dim, x_index_exp_iter):
    all_nodes_coords = [(np.array(coord, dtype=float), index) for coord, index in all_nodes_coords]
    
    w = weight_function(all_nodes_coords)
    
    w.spatial_dim = spatial_dim
    
    
    for x, index, exp in x_index_exp_iter:
        x = np.array(x, dtype=float)
        exp = np.array(exp, dtype=float)
        w.partial_order = 1
        act = w(x, index)
        assert_almost_equal(np.linalg.norm(exp - act), 0)
        w.partial_order = 0
        act2 = w(x, index)
        eq_(act[0], act2[0])
        eq_(act2.shape, (1,))
        
def test_poly_kernel_1d():
    spatial_dim = 1
    x = -2.5
    order_partial_order_exp_iter = [
        (1, 0, [1, -2.5]),
        (2, 0, [1, -2.5, 6.25]),
        (3, 0, [1, -2.5, 6.25, -15.625]),
        (1, 1, [[1, -2.5], [0, 1]]),
        (2, 1, [[1, -2.5, 6.25], [0, 1, -5]]),
        (3, 1, [[1, -2.5, 6.25, -15.625], [0, 1, -5, 18.75]])
        ]
    _test_poly_kernel(spatial_dim, x, order_partial_order_exp_iter)

def test_poly_kernel_2d():
    spatial_dim = 2
    x = np.array((-2.5, -3))
    
    order_partial_order_exp_iter = [
        (1, 0, [1, -2.5, -3]),
        (2, 0, [1, -2.5, -3, 6.25, 7.5, 9]),
        (3, 0, [1, -2.5, -3, 6.25, 7.5, 9, -15.625, -18.75, -22.5, -27]),
        (1, 1, [[1, -2.5, -3],
                [0, 1, 0],
                [0, 0, 1]]),
        (2, 1, [[1, -2.5, -3, 6.25, 7.5, 9],
                [0, 1, 0, -5, -3, 0],
                [0, 0, 1, 0, -2.5, -6]]),
        (3, 1, [[1, -2.5, -3, 6.25, 7.5, 9, -15.625, -18.75, -22.5, -27],
                [0, 1, 0, -5, -3, 0, 18.75, 15, 9, 0],
                [0, 0, 1, 0, -2.5, -6, 0, 6.25, 15, 27]])]
    
    _test_poly_kernel(spatial_dim, x, order_partial_order_exp_iter)

def _test_poly_kernel(spatial_dim, x, order_partial_order_exp_iter):
    kernel = PolynomialKernel()
    kernel.spatial_dim = spatial_dim
    
    sf = order_partial_order_exp_iter.copy()
    random.shuffle(sf)
    order_partial_order_exp_iter.extend(sf)
    
    for order, partial_order, exp in order_partial_order_exp_iter:
        kernel.order = order
        kernel.partial_order = partial_order
        act = kernel(x)
        act = np.array(act, dtype=float)
        assert_almost_equal(np.linalg.norm(act - exp), 0)

def test_trans_coords():
    coords=np.array([
         (0,0),
         (0,1),
         (0,0.5),
         (0.5,0),
         (0.5,0.5),
         (0.7,0.3)           
         ])
    for dst_range,src_range,exps in [
               (np.array([[0,0],[-2,1.5]]),None,
                    np.array([
                          (0,0),
                          (0,1.5),
                          (0,0.75),
                          (-1,0),
                          (-1,0.75),
                          (-1.4,0.45)
                          ])),
                (np.array([[1,-0.5],[-3,1.5]]),None,
                    np.array([
                          (1,-0.5),
                          (1,1.5),
                          (1,0.5),
                          (-1,-0.5),
                          (-1,0.5),
                          (-1.8,0.1)
                              ])
                 ),
                 (np.array([[1,-0.5],[-3,1.5]]),np.array([[0.5,0.5],[-1.5,-2.5]]),
                    np.array([
                              (0,-0.5+1/3),
                              (0,-0.5-2/6),
                              (0,-0.5),
                              (1,-0.5+2/6),
                              (1,-0.5),
                              (1.4,-0.5+2/15)
                              ])
                  ),
                                      
        ]:
        acts=trans_coords(coords,dst_range,src_range)
        assert_almost_equal(0,np.linalg.norm(exps-acts))
    
if __name__ == '__main__':
    import nose
    nose.main()
