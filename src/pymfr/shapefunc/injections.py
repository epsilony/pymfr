'''

@author: "epsilonyuan@gmail.com"
'''
from injector import Module, provides, inject, Key, ClassProvider, InstanceProvider
from pymfr.shapefunc.shape_func import CoredShapeFunction, ShapeFunction, ShapeFunctionCore, RawCoordRadiusGetter
from pymfr.shapefunc.mlsrk import MLSRK
from pymfr.misc.weight_func import WeightFunction, RegularNodeRadiusBasedDistanceFunction, triple_spline
from pymfr.misc.kernel import PolynomialKernel

ShapeWeightFunction = Key('shape_weight_function')
ShapeWeightFunctionCore = Key('shape_weight_function_core')
ShapeKernelFunction = Key('shape_kernel_function')

class MLSRKShapeFunctionModule(Module):

    shape_weight_function_core_provider = InstanceProvider(triple_spline)
    
    def __init__(self, kernel_order=1):
        self.kernel_order = 1
    
    def configure(self, binder):
        binder.bind(ShapeWeightFunctionCore, to=self.shape_weight_function_core_provider)
    
    @provides(ShapeFunction)
    @inject(shape_function_core=ShapeFunctionCore)
    def shape_function(self, shape_function_core):
        return CoredShapeFunction(shape_function_core)
    
    @provides(ShapeFunctionCore)
    @inject(weight_func=ShapeWeightFunction,
            kernel=ShapeKernelFunction
            )
    def shape_function_core(self, weight_func, kernel):
        return MLSRK(weight_func, kernel)

    @provides(ShapeWeightFunction)
    @inject(core=ShapeWeightFunctionCore)
    def shape_weight_function(self, core):
        coord_rad_getter = RawCoordRadiusGetter()
        node_regular_dist_func = RegularNodeRadiusBasedDistanceFunction(coord_rad_getter)
        return WeightFunction(node_regular_dist_func, core)

    @provides(ShapeKernelFunction)
    def shape_kernel_Function(self):
        ret = PolynomialKernel()
        ret.order = self.kernel_order
        return ret
        
