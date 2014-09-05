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


class Test(unittest.TestCase):

    def setUp(self):
        pass


    def tearDown(self):
        pass


    def test_partial_of_unity(self):
        
        xs=[(-0.2,),(0.7,)]
        coords=self._oned_uniform_coords()
        radius=self._oned_uniform_radius()
        spatial_dim=1
        
        kernel=PolynomialKernel()
        kernel.order=1
        mlsrk=self.mlsrk(coords, radius, kernel)
        mlsrk.spatial_dim=spatial_dim
        node_indes = [i for i in range(len(coords))]
        for x in xs:
            x=np.array(x,dtype=float)
            mlsrk.patial_order=1
            act=mlsrk.calc(x, coords, node_indes)
            self.assertAlmostEqual(0, np.linalg.norm(act.sum(axis=1)-(1,0)))
    
    def mlsrk(self,coords,radius,kernel):
        return MLSRK(self.weight_func(coords,radius),kernel)
    
    def weight_func(self,coords,radius):
        coord_rad_getter=lambda index:(coords[index],radius[index])
        node_regular_dist_func=InfluencRadiusBasedNodeRegularDistance(coord_rad_getter)
        return WeightFunction(node_regular_dist_func)        

    def _oned_uniform_coords(self):
        return np.linspace(-2, 3, 11).reshape((-1,1))
    
    def _oned_uniform_radius(self):
        res= np.empty((11,1))
        res.fill(1.25)
        return res

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()