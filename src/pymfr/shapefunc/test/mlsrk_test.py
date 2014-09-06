'''

@author: "epsilonyuan@gmail.com"
'''

from unittest import TestCase
import numpy as np
import unittest
from pymfr.misc.weight_func import WeightFunction, InfluencRadiusBasedNodeRegularDistance
from pymfr.shapefunc.mlsrk import MLSRK
from pymfr.shapefunc import mlsrk
from pymfr.misc.kernel import PolynomialKernel
import random


class MLSRK_Test(unittest.TestCase):

    def setUp(self):
        pass


    def tearDown(self):
        pass


    def test_partial_of_unity(self):
        xs=[(-0.2,),(0.7,),(-1.8,),(2.9,)]
        coords=self._oned_uniform_coords()        
        spatial_dim=1

        for order,r in (1,1.25),(2,1.25),(3,3):
            radius=self._oned_uniform_radius(r)
            self._test_partial_of_unit(xs, coords, radius, spatial_dim, order)
    
    def test_poly_kernel_linear_tran_invar(self):
        xs=[(-0.1,),(0.6,),(-1.9,),(2.85,)]
        coords=self._oned_uniform_coords()        
        spatial_dim=1

        for order,r in (1,1.25),(2,1.5),(3,3):
            radius=self._oned_uniform_radius(r)
            self._test_poly_kernel_linear_trans_invar(xs, coords, radius, spatial_dim, order)
    
    def _test_poly_kernel_linear_trans_invar(self,xs,coords,radius,spatial_dim,order):
        kernel=PolynomialKernel()
        kernel.order=order
        kernel2=PolynomialKernel()
        kernel2._is_linear_transformable=False
        kernel2.order=order
        
        mlsrk=self.create_mlsrk(coords, radius, kernel)
        mlsrk.spatial_dim=spatial_dim
        
        mlsrk2=self.create_mlsrk(coords,radius,kernel2)
        mlsrk2.spatial_dim=spatial_dim
        
        node_indes = [i for i in range(len(coords))]
        
        xs2=xs.copy()
        random.shuffle(xs2)
        xs.extend(xs2)
        for x in xs:
            x=np.array(x,dtype=float)
            mlsrk.partial_order=1
            filtered_indes=[i for i in node_indes if np.linalg.norm(x-coords[i])<radius[i]]
            inputed_coords=coords[filtered_indes]
            act=mlsrk.calc(x, inputed_coords, filtered_indes)
            act2=mlsrk2.calc(x,inputed_coords,filtered_indes)
            self.assertAlmostEqual(0, np.linalg.norm(act-act2))
    
    def _test_partial_of_unit(self,xs,coords,radius,spatial_dim,order):
        kernel=PolynomialKernel()
        kernel.order=order
        mlsrk=self.create_mlsrk(coords, radius, kernel)
        mlsrk.spatial_dim=spatial_dim
        node_indes = [i for i in range(len(coords))]
        
        xs2=xs.copy()
        random.shuffle(xs2)
        xs.extend(xs2)
        for x in xs:
            x=np.array(x,dtype=float)
            mlsrk.partial_order=1
            filtered_indes=[i for i in node_indes if np.linalg.norm(x-coords[i])<radius[i]]
            inputed_coords=coords[filtered_indes]
            act=mlsrk.calc(x, inputed_coords, filtered_indes)
            self.assertAlmostEqual(0, np.linalg.norm(act.sum(axis=1)-(1,0)))
    
    def create_mlsrk(self,coords,radius,kernel):
        return MLSRK(self.weight_func(coords,radius),kernel)
    
    def weight_func(self,coords,radius):
        coord_rad_getter=lambda index:(coords[index],radius[index])
        node_regular_dist_func=InfluencRadiusBasedNodeRegularDistance(coord_rad_getter)
        return WeightFunction(node_regular_dist_func)        

    def _oned_uniform_coords(self):
        return np.linspace(-2, 3, 11).reshape((-1,1))
    
    def _oned_uniform_radius(self,radius=1.25):
        res= np.empty((11,1))
        res.fill(radius)
        return res

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()