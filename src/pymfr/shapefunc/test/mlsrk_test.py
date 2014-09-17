'''

@author: "epsilonyuan@gmail.com"
'''

import random
import unittest

import numpy as np
from pymfr.misc.kernel import PolynomialKernel
from pymfr.misc.weight_func import WeightFunction, RegularNodeRadiusBasedDistanceFunction
from pymfr.shapefunc.mlsrk import MLSRK
import sympy as sm
from pymfr.misc.tools import twod_uniform_coords


class MLSRK_Test(unittest.TestCase):

    def setUp(self):
        pass


    def tearDown(self):
        pass

    def test_poly_fit_1d(self):
        coords = self._oned_uniform_coords(-3, 13, 17)
        rads = np.empty(len(coords))
        spatial_dim = 1
        
        kernel = PolynomialKernel()
        x = sm.symbols('x')
        
        p1 = 10 - 2 * x
        p2 = 11.1 - 1.9 * x
        p3 = -3.1 + 2.3 * x - 1.1 * x * x
        p4 = -1.7 + 1.1 * x + 0.7 * x * x - 0.34 * x ** 3
        
        pts = np.linspace(-2.9, 12.9, 10)
        pts = pts.reshape((-1, 1))
        
        for order, rad, poly in [(1, 1.1, p1), (1, 1.2, p2), (2, 2.1, p3), (2, 2.2, p2), (3, 3.1, p4), (3, 3.2, p3), (3, 3.2, p2)]:
            rads.fill(rad)
            kernel.order = order
            self._test_poly_fit(poly, pts, coords, rads, kernel, spatial_dim)
    
    def test_poly_fit_2d(self):
        
        xs = np.linspace(-1, 9, 11)
        ys = np.linspace(-2.5, 8.5, 11)
        coords = twod_uniform_coords(xs, ys)
        rads = np.empty(len(coords))
        spatial_dim = 2
        
        kernel = PolynomialKernel()
        x, y = sm.symbols('x,y')
        
        p1 = 10 - 2 * x + y
        p2 = 11.1 - 1.9 * x + 1.1 * y
        p3 = -3.1 + 2.3 * x - 2 * y - 1.1 * x * x + 0.7 * x * y - 0.3 * y * y
        p4 = p3 + 1.1 * x ** 3 - 0.2 * x ** 2 * y + 0.9 * x * y ** 2 + 0.2 * y ** 3
        
        pts_xs = np.linspace(-0.7, 8.9, 3)
        pts_ys = np.linspace(-2.3, 8.1, 3)
        pts = twod_uniform_coords(pts_xs, pts_ys)
        
        for order, rad, poly in [(1, 1.2, p1), (1, 1.3, p2), (2, 2.2, p3), (2, 2.2, p2), (3, 3.5, p4), (3, 3.5, p3), (3, 3.4, p2)]:
            rads.fill(rad)
            kernel.order = order
            self._test_poly_fit(poly, pts, coords, rads, kernel, spatial_dim)
    
    def _test_poly_fit(self, poly, pts, coords, rads, kernel, spatial_dim):
        exp_func = self._lambda_partial_poly(poly, spatial_dim)
        mlsrk = self.create_mlsrk(coords, rads, kernel)
        mlsrk.spatial_dim = spatial_dim
        mlsrk.partial_order = 1
        node_indes = [i for i in range(len(coords))]
        node_values = np.array([exp_func(coord)[0] for coord in coords], dtype=float)
        for pt in pts:
            mlsrk.partial_order = 1
            filtered_indes = [i for i in node_indes if np.linalg.norm(pt - coords[i]) < rads[i]]
            inputed_coords = coords[filtered_indes]
            sf_val = mlsrk.calc(pt, inputed_coords, filtered_indes)
            exp = exp_func(pt)    
            ts = np.empty(sf_val.shape)
            filtered_node_values = node_values[filtered_indes]
            ts = sf_val * filtered_node_values.reshape(-1)
            act = ts.sum(axis=1)
            self.assertAlmostEqual(0, np.linalg.norm(exp - act))
        
    
    def _lambda_partial_poly(self, poly, spatial_dim):
        polys = [poly]
        xyz = 'x', 'y', 'z'
        polys.extend([poly.diff(r) for r in xyz[:spatial_dim]])
        
        def _f(pt):
            subs_args = [(r, pt[i]) for r, i in zip(xyz[:spatial_dim], range(spatial_dim))]
            return np.array([p.subs(subs_args) for p in polys], dtype=float)
        
        return _f

    def test_partial_of_unity_uni_1d(self):
        xs = [(-0.2,), (0.7,), (-1.8,), (2.9,)]
        nodes_size = 11
        nodes_start = -2
        nodes_end = 3
        coords = self._oned_uniform_coords(nodes_start, nodes_end, nodes_size)        
        spatial_dim = 1

        for order, r in (1, 1.25), (2, 1.25), (3, 3):
            radius = self._oned_uniform_radius(nodes_size, r)
            self._test_partial_of_unity(xs, coords, radius, spatial_dim, order)
    
    def test_partial_of_unity_uni_2d(self):
        xs = np.linspace(-1, 9, 9)
        ys = np.linspace(3, 13, 9)
        coords = twod_uniform_coords(xs, ys)
        
        rads = np.empty(len(coords))
        
        pt_xs = np.linspace(-0.8, 8.8, 7)
        pt_ys = np.linspace(3.3, 12.6, 7)
        pts = twod_uniform_coords(pt_xs, pt_ys)
        
        spatial_dim = 2
        
        for order, r in (1, 1.4), (2, 2.6), (3, 5.1):
            rads.fill(r)
            self._test_partial_of_unity(pts, coords, rads, spatial_dim, order)
        
    
    def test_poly_kernel_linear_tran_invar_1d(self):
        xs = [(-0.1,), (0.6,), (-1.9,), (2.85,)]
        nodes_size = 11
        nodes_start = -2
        nodes_end = 3
        coords = self._oned_uniform_coords(nodes_start, nodes_end, nodes_size)        
        spatial_dim = 1

        for order, r in (1, 1.25), (2, 1.5), (3, 3):
            radius = self._oned_uniform_radius(nodes_size, r)
            self._test_poly_kernel_linear_trans_invar(xs, coords, radius, spatial_dim, order)
    
    def _test_poly_kernel_linear_trans_invar(self, xs, coords, radius, spatial_dim, order):
        kernel = PolynomialKernel()
        kernel.order = order
        kernel2 = PolynomialKernel()
        kernel2._is_linear_transformable = False
        kernel2.order = order
        
        mlsrk = self.create_mlsrk(coords, radius, kernel)
        mlsrk.spatial_dim = spatial_dim
        
        mlsrk2 = self.create_mlsrk(coords, radius, kernel2)
        mlsrk2.spatial_dim = spatial_dim
        
        node_indes = [i for i in range(len(coords))]
        
        xs2 = xs.copy()
        random.shuffle(xs2)
        xs.extend(xs2)
        for x in xs:
            x = np.array(x, dtype=float)
            mlsrk.partial_order = 1
            filtered_indes = [i for i in node_indes if np.linalg.norm(x - coords[i]) < radius[i]]
            inputed_coords = coords[filtered_indes]
            act = mlsrk.calc(x, inputed_coords, filtered_indes)
            act2 = mlsrk2.calc(x, inputed_coords, filtered_indes)
            self.assertAlmostEqual(0, np.linalg.norm(act - act2))
    
    def _test_partial_of_unity(self, pts, coords, radius, spatial_dim, order):
        kernel = PolynomialKernel()
        kernel.order = order
        mlsrk = self.create_mlsrk(coords, radius, kernel)
        mlsrk.spatial_dim = spatial_dim
        node_indes = [i for i in range(len(coords))]
        
        xs2 = pts.copy()
        random.shuffle(xs2)
        
        if not isinstance(pts, list):
            pts = list(pts)
        pts.extend(xs2)
        exp_sum = np.zeros(spatial_dim + 1)
        exp_sum[0] = 1
        for pt in pts:
            pt = np.array(pt, dtype=float)
            mlsrk.partial_order = 1
            filtered_indes = [i for i in node_indes if np.linalg.norm(pt - coords[i]) < radius[i]]
            inputed_coords = coords[filtered_indes]
            act = mlsrk.calc(pt, inputed_coords, filtered_indes)
            act_sum = act.sum(axis=1)
            # print("act_sum: %s"+str(act_sum))
            self.assertAlmostEqual(0, np.linalg.norm(act_sum - exp_sum))
    
    def create_mlsrk(self, coords, radius, kernel):
        return MLSRK(self.weight_func(coords, radius), kernel)
    
    def weight_func(self, coords, radius):
        coord_rad_getter = lambda index:(coords[index], radius[index])
        node_regular_dist_func = RegularNodeRadiusBasedDistanceFunction(coord_rad_getter)
        return WeightFunction(node_regular_dist_func)        

    def _oned_uniform_coords(self, start, end, num):
        return np.linspace(start, end, num).reshape((-1, 1))
    
    def _oned_uniform_radius(self, num, radius=1.25):
        res = np.empty((num, 1))
        res.fill(radius)
        return res
    
if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
