'''

@author: "epsilonyuan@gmail.com"
'''

from injector import Module, provides, inject, Key, ClassProvider, singleton
from pymfr.model.searcher import SupportNodeSearcher, SegmentLagrangleSupportNodeSearcher, \
    NodeLagrangleSupportNodeSearcher
from pymfr.shapefunc.shape_func import ShapeFunction, LinearShapeFuntion, SoloShapeFunction
from pymfr.process.load import LoadCalculator, SimpLoadCalculator
from pymfr.process.process import SimpProcessorCore, SimpAssemblersConsumer, LagrangleDirichletProcessorCore, \
    SimpLagrangleDirichletConsumer, SimpProcessor
from pymfr.process.assembler import VirtualLoadWorkAssembler, LagrangleDirichletLoadAssembler, PoissonVolumeAssembler
from pymfr.process.post import SimpPostProcessor


VolumeProcessorCore = Key('volume_processor_core')
NeumannProcessorCore = Key('neumann_processor_core')
DirichletProcessorCore = Key('dirichlet_processor_core')

VolumeProcessConsumer = Key('volume_process_consumer')
NeumannProcessConsumer = Key('neumann_process_consumer')
DirichletProcessConsumer = Key('dirichlet_process_consumer')

VolumeAssembler = Key('volume_assembler')
VolumeLoadAssembler = Key('volume_load_assmbler')
NeumannAssembler = Key('neumann_assembler')
DirichletAssembler = Key('dirichlet_assembler')

LagrangleSupportNodeSearcher = Key('lagrangle_support_node_searcher')
LagrangleShapeFunction = Key('lagrangle_shape_function')


class SimpProcessorModule(Module):
    
    load_calculator_cls = SimpLoadCalculator
    
    def configure(self, binder):
        binder.bind(LoadCalculator, to=ClassProvider(self.load_calculator_cls), scope=singleton)
    @provides(SimpProcessor)
    @inject(
            volume_core=VolumeProcessorCore,
            neumann_core=NeumannProcessorCore,
            dirichlet_core=DirichletProcessorCore,
            )
    def simp_processor(self, **kwargs):
        return SimpProcessor(**kwargs)

class SimpVolumeNeumannProcessorModule(Module):
    
    @provides(VolumeProcessorCore)
    @inject(support_node_searcher=SupportNodeSearcher,
            shape_func=ShapeFunction,
            load_calculator=LoadCalculator,
            consumer=VolumeProcessConsumer,
            )
    def volume_processor_core(self, **kwargs):
        return SimpProcessorCore(**kwargs)
    
    @provides(VolumeProcessConsumer)
    @inject(volume_assembler=VolumeAssembler,
            volume_load_assembler=VolumeLoadAssembler
            )
    def volume_process_consumer(self, volume_assembler, volume_load_assembler):
        return SimpAssemblersConsumer(volume_assembler, volume_load_assembler)
    
    @provides(VolumeLoadAssembler)
    def volume_load_assembler(self):
        return VirtualLoadWorkAssembler()
    
    @provides(NeumannProcessorCore)
    @inject(support_node_searcher=SupportNodeSearcher,
            shape_func=ShapeFunction,
            load_calculator=LoadCalculator,
            consumer=NeumannProcessConsumer,
            )
    def neumann_processor_core(self, **kwargs):
        return SimpProcessorCore(**kwargs)
    
    @provides(NeumannProcessConsumer)
    @inject(assembler=NeumannAssembler)
    def neumann_process_consumer(self, assembler):
        return SimpAssemblersConsumer(None, assembler)
    
    @provides(NeumannAssembler)
    def neumann_assembler(self):
        return VirtualLoadWorkAssembler()

class LagrangleDirichletProcessorModule(Module):
    @provides(DirichletProcessorCore)
    @inject(
             support_node_searcher=SupportNodeSearcher,
             shape_func=ShapeFunction,
             lagrangle_support_node_searcher=LagrangleSupportNodeSearcher,
             lagrangle_shape_func=LagrangleShapeFunction,
             load_calculator=LoadCalculator,
             consumer=DirichletProcessConsumer
            )
    def dirichlet_processor_core(self, **kwargs):
        return LagrangleDirichletProcessorCore(**kwargs)
    
    @provides(DirichletProcessConsumer)
    @inject(assembler=DirichletAssembler)
    def dirichlet_process_consumer(self, assembler):
        return SimpLagrangleDirichletConsumer(assembler)
    
    
    @provides(DirichletAssembler)
    def dirichlet_assembler(self):
        return LagrangleDirichletLoadAssembler()
    
class PoissonVolumeAssemblerModule(Module):
    
    @provides(VolumeAssembler)
    def volume_assembler(self):
        return PoissonVolumeAssembler()

class LagrangleCommon2D(Module):
    @provides(LagrangleShapeFunction)
    def lagrangle_shape_function(self):
        return LinearShapeFuntion()
    
    @provides(LagrangleSupportNodeSearcher)
    def lagrangle_support_node_searcher(self):
        return SegmentLagrangleSupportNodeSearcher()

class LagrangleCommon1D(Module):
    @provides(LagrangleShapeFunction)
    def lagrangle_shape_function(self):
        return SoloShapeFunction()
    
    @provides(LagrangleSupportNodeSearcher)
    def lagrangle_support_node_seacher(self):
        return NodeLagrangleSupportNodeSearcher()

def get_simp_poission_processor_modules(spatial_dim=2):
    ret = [SimpProcessorModule,
            SimpVolumeNeumannProcessorModule,
            LagrangleDirichletProcessorModule,
            PoissonVolumeAssemblerModule
            ]
    lags = {1:LagrangleCommon1D, 2:LagrangleCommon2D}
    ret.append(lags[spatial_dim])
    return ret

class SimpPostProcessorModule(Module):
    
    @provides(SimpPostProcessor)
    @inject(support_node_searcher=SupportNodeSearcher,
            shape_func=ShapeFunction,
            )
    def simp_post_processor(self, **kwargs):
        return SimpPostProcessor(**kwargs)
