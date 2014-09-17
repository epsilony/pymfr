'''

@author: "epsilonyuan@gmail.com"
'''
from pymfr.misc.mixin import SetupMixin

class SimpProcessorCore:
    
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
    
    def process(self, quadrature_points):
        for quadrature_point in quadrature_points:
            node_indes = self.support_node_searcher.search_indes(
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
            (load, load_validity) = (None, None) if load_val is None else load_val
            self.consumer(quadrature_point.weight,
                          node_indes,
                          shape_func_value,
                          load,
                          load_validity)


class LagrangleDirichletProcessorCore(SetupMixin):
    
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
    
    def process(self, quadrature_points):
        for quadrature_point in quadrature_points:
            node_indes = self.support_node_searcher.search_indes(
                                quadrature_point.coord,
                                quadrature_point.bnd)
            shape_func_value = self.shape_func(
                               quadrature_point.coord,
                               node_indes)
            lagrangle_node_indes = self.lagrangle_support_node_searcher.search_indes(
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
            (load, load_validity) = (None, None) if load_val is None else load_val

            self.consumer(quadrature_point.weight,
                          node_indes,
                          shape_func_value,
                          load,
                          load_validity,
                          lagrangle_node_indes,
                          lagrangle_shape_func_value)


class SimpAssemblersConsumer:
    def __init__(self, volume_assembler=None, load_assembler=None):
        self.volume_assembler = volume_assembler
        self.load_assembler = load_assembler
    
    def __call__(self, weight, node_indes, shape_func_value, load, load_validity):
        if self.volume_assembler is not None:
            self.volume_assembler.assemble(weight, node_indes, shape_func_value)
        if self.load_assembler is not None and load is not None:
            self.load_assembler.assemble(weight, node_indes, shape_func_value, load=load)

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


class SimpProcessor(SetupMixin):
    
    __prerequisites__ = ['volume_quadrature_points',
                         'neumann_quadrature_points',
                         'dirichlet_quadrature_points']
    
    def __init__(self,
                 volume_core, neumann_core, dirichlet_core,
                 volume_quadrature_points=None,
                 neumann_quadrature_points=None,
                 dirichlet_quadrature_poins=None
                 ):
        self.volume_core = volume_core
        self.neumann_core = neumann_core
        self.dirichlet_core = dirichlet_core
        self.volume_quadrature_points = volume_quadrature_points
        self.neumann_quadrature_points = neumann_quadrature_points
        self.dirichlet_quadrature_points = dirichlet_quadrature_poins
    
    def process(self):
        for core, pts in zip((self.volume_core, self.neumann_core, self.dirichlet_core),
                        (self.volume_quadrature_points,
                         self.neumann_quadrature_points,
                         self.dirichlet_quadrature_points)):
            if not core or not pts:
                continue
            core.process(pts)


