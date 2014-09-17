'''

@author: "epsilonyuan@gmail.com"
'''
from injector import Module, provides, inject, Key, ClassProvider
from pymfr.shapefunc.shape_func import CoredShapeFunction, ShapeFunction, ShapeFunctionCore, RawCoordRadiusGetter
from pymfr.shapefunc.mlsrk import MLSRK
from pymfr.misc.weight_func import WeightFunction, RegularNodeRadiusBasedDistanceFunction, TripleSpline

ShapeWeightFunction = Key('shape_weight_function')
ShapeWeightFunctionCore = Key('shape_weight_function_core')
ShapeKernelFunction = Key('shape_kernel_function')

class MLSRKShapeFunctionModule(Module):

    shape_weight_function_core_cls = TripleSpline
    
    def configurature(self, binder):
        binder.bind(ShapeWeightFunctionCore, to=ClassProvider(self.shape_weight_function_core_cls))
    
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
