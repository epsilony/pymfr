'''

@author: "epsilonyuan@gmail.com"
'''
from pymfr.misc.mixin import SetupMixin
import logging

class SimpProcessorCore(SetupMixin):
    __prerequisites__ = ['quadrature_points']
    
    def __init__(self,
                 support_node_searcher,
                 shape_func,
                 load_calculator,
                 consumer
                 ):
        self.support_node_searcher = support_node_searcher
        self.load_calculator = load_calculator
        self.shape_func = shape_func
        self.consumer = consumer
    
    def process(self):
        for quadrature_point in self.quadrature_points:
            node_indes = self.support_node_searcher(
                                quadrature_point.coord,
                                quadrature_point.bnd)
            shape_func_value = self.shape_func(
                               quadrature_point.coord,
                               node_indes)
            load_val = self.load_calculator(
                                            quadrature_point.load_key,
                                            quadrature_point.coord,
                                            quadrature_point.bnd
                                              )
            load, load_validity = None, None if load_val is None else load_val
            self.consumer(quadrature_point.weight,
                          node_indes,
                          shape_func_value,
                          load,
                          load_validity)


class LagrangleDirichletProcessorCore(SetupMixin):
    
    __prerequisites__ = ['quadrature_points']
    
    def __init__(self,
                 support_node_searcher,
                 shape_func,
                 lagrangle_support_node_searcher,
                 lagrangle_shape_func,
                 load_calculator,
                 consumer
                 ):
        self.support_node_searcher = support_node_searcher
        self.load_calculator = load_calculator
        self.shape_func = shape_func
        self.lagrangle_support_node_searcher = lagrangle_support_node_searcher
        self.lagrangle_shape_func = lagrangle_shape_func
        self.consumer = consumer
    
    def process(self):
        for quadrature_point in self.quadrature_points:
            node_indes = self.support_node_searcher(
                                quadrature_point.coord,
                                quadrature_point.bnd)
            shape_func_value = self.shape_func(
                               quadrature_point.coord,
                               node_indes)
            lagrangle_node_indes = self.lagrangle_supoort_domain_searcher(
                                 quadrature_point.coord,
                                 quadrature_point.bnd)
            lagrangle_shape_func_value = self.lagrangle_shape_func(
                                 quadrature_point.coord,
                                 lagrangle_node_indes)
            load_val = self.load_calculator(
                                            quadrature_point.load_key,
                                            quadrature_point.coord,
                                            quadrature_point.bnd
                                              )
            load, load_validity = None, None if load_val is None else load_val
            if load_validity is not None:
                logging.warning('not None validity of load {}({},{})',
                                quadrature_point.load_key,
                                quadrature_point.coord,
                                quadrature_point.bnd
                                )
            self.consumer(quadrature_point.weight,
                          node_indes,
                          shape_func_value,
                          load,
                          load_validity,
                          lagrangle_node_indes,
                          lagrangle_shape_func_value)


class SimpAssemblersConsumer:
    def __init__(self, assemblers):
        self.assembles = assemblers
    
    def __call__(self, weight, node_indes, shape_func_value, load, load_validity):
        for assembler in self.assemblers:
            assembler.assemble(weight, node_indes, shape_func_value, load)

class SimpDirichletConsumer:
    def __init__(self, assembler):
        self.assembler = assembler
        
    def __call__(self, weight, node_indes, shape_func_value, load, load_validity):
        self.assembler.assemble(weight, node_indes, shape_func_value, load, load_validity)

class SimpLagrangleDirichletConsumer:
    def __init__(self, assembler):
        self.assembler = assembler
    
    def __call__(self,
                 weight, node_indes, shape_func_value,
                 load, load_validity,
                 lagrangle_node_idnes, lagrangle_shape_func_value):
        self.assembler.assemble(
                                weight, node_indes, shape_func_value,
                                load, load_validity,
                                lagrangle_node_idnes, lagrangle_shape_func_value
                                )


class SimpProcessor:
    
    def __init__(self, volume_core, neumann_core, dirichlet_core):
        self.volume_core = volume_core
        self.neumann_core = neumann_core
        self.dirichlet_core = dirichlet_core
        
    def process(self):
        for core in (self.volume_core, self.neumann_core, self.dirichlet_core):
            core.process()


