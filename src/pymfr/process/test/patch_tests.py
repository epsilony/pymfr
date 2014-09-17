'''

@author: "epsilonyuan@gmail.com"
'''
from math import pi, sin, cos
import numpy as np
from pymfr.model.raw_model import MFNode, OneDModel
from injector import Injector
from pymfr.shapefunc.injections import MLSRKShapeFunctionModule
from pymfr.model.injections import OneDSupportNodesSearcherModule
from pymfr.process.injections import get_simp_poission_processor_modules, SimpPostProcessorModule
from pymfr.process.process import SimpProcessor
from pymfr.process.quadrature import iter_quadrature_units
from pymfr.misc.tools import search_setup_mixins, gen_setup_status
from pymfr.process.post import SimpPostProcessor

class PoissonOneD:
    
    cases = [
           {'volume_load':lambda coord, bnd:(np.zeros(1, dtype=float), None),
            'dirichlet_left':lambda coord, bnd:(np.zeros(1, dtype=float), (True,)),
            'dirichlet_right':lambda coord, bnd:(np.ones(1, dtype=float), (True,)),
            'exp':lambda coord:np.array((coord[0], 0), dtype=float)
            },
          {'volume_load':lambda coord, bnd:(np.array((8,), dtype=float), None),
            'dirichlet_left':lambda coord, bnd:(np.zeros(1, dtype=float), (True,)),
            'dirichlet_right':lambda coord, bnd:(np.ones(1, dtype=float), (True,)),
            'exp':lambda coord:np.array((-4 * coord[0] ** 2 + 5 * coord[0],
                                        - 8 * coord[0] + 5), dtype=float)
            },
          {'volume_load':lambda coord, bnd:(8 + 16 * coord, None),
            'dirichlet_left':lambda coord, bnd:(np.zeros(1, dtype=float), (True,)),
            'dirichlet_right':lambda coord, bnd:(np.ones(1, dtype=float), (True,)),
            'exp':lambda coord:np.array((-8 / 3 * coord[0] ** 3 - 4 * coord[0] ** 2 + 23 / 3 * coord[0],
                                         - 8 * coord[0] ** 2 - 8 * coord[0] + 23 / 3
                                         ), dtype=float)
            },
          {'volume_load':lambda coord, bnd:(4 * pi ** 2 * np.sin(2 * pi * coord), None),
            'dirichlet_left':lambda coord, bnd:(np.zeros(1, dtype=float), (True,)),
            'dirichlet_right':lambda coord, bnd:(np.zeros(1, dtype=float), (True,)),
            'exp':lambda coord:np.array((sin(2 * pi * coord[0]),
                                              2 * pi * cos(2 * pi * coord[0])))
            },
           ]
    
    def __init__(self, case=2, nodes=None):
        self.nodes = nodes
        self.case = case
    
    def set_nodes_by_num(self, num):
        xs = np.linspace(0, 1, num)
        nodes = [MFNode(np.array((x,), dtype=float)) for x in xs]
        self.nodes = nodes
    
    def set_nodes_radiuses(self, radiuses):
        if np.isscalar(radiuses):
            for node in self.nodes:
                node.radius = radiuses
        else:
            if len(radiuses) != len(self.nodes):
                raise ValueError()
            for i, node in enumerate(self.nodes):
                node.radius = radiuses[i]
    
    def index_nodes(self):
        nodes = self.nodes
        for i, node in enumerate(nodes):
            node.index = i
        
        left_dirichlet_node = nodes[0]
        right_dirichlet_node = nodes[-1]        
        left_dirichlet_node.lagrangle_index = len(nodes)
        right_dirichlet_node.lagrangle_index = len(nodes) + 1
    
    def gen_model(self):
        self.index_nodes()
        nodes = self.nodes
        case_data = self.cases[self.case]
        model = OneDModel()
        model.nodes = nodes
        
        left_dirichlet_node = nodes[0]
        right_dirichlet_node = nodes[-1]
        model.dirichlet_nodes = [left_dirichlet_node, right_dirichlet_node]
        model.add_dirichlet_load(left_dirichlet_node, case_data['dirichlet_left'])
        model.add_dirichlet_load(right_dirichlet_node, case_data['dirichlet_right'])
        
        model.set_volume_load(case_data['volume_load'])
        
        model.gen_quadrature_units()
        return model
        
def get_oned_poisson_patch_injector():
    injector = Injector()
    modules = [MLSRKShapeFunctionModule, OneDSupportNodesSearcherModule]
    modules.extend(get_simp_poission_processor_modules(spatial_dim=1))
    for mod in modules:
        injector.binder.install(mod)
    return injector

def test_poisson_1d():
    basic_data = {'nodes_num':21,
                'value_dim':1,
                'spatial_dim':1,
                'lagrangle_nodes_size':2,
                'radius_ratio':2.5,
                'error_lim':4e-3 #max_abs_diff/max_abs_exp
                }

    datas = []
    for i in range(4):
        data = {}
        data.update(basic_data)
        data['case'] = i
        data['quadrature_degree'] = i + 1
        data['kernel_order'] = i + 1
        datas.append(data)
    
    for data in datas:
        _test_poisson_1d(**data)
        


def _test_poisson_1d(case, error_lim, kernel_order, quadrature_degree, nodes_num, radius_ratio, value_dim, spatial_dim, lagrangle_nodes_size):
    vector_size = (nodes_num + lagrangle_nodes_size) * value_dim
    
    
    injector = get_oned_poisson_patch_injector()
    injector.binder.install(SimpPostProcessorModule)
    processor = injector.get(SimpProcessor)
    print(processor)
    p1d = PoissonOneD()
    p1d.case = case
    p1d.set_nodes_by_num(nodes_num)
    model = p1d.gen_model()
    model.volume_quadrature_degree = quadrature_degree
    model.gen_quadrature_units()
    

    common_data = {'vector':np.zeros(vector_size, dtype=float),
                 'matrix':np.zeros((vector_size, vector_size), dtype=float),
                 'lagrangle_nodes_size':lagrangle_nodes_size,
                 'value_dim':value_dim,
                 'spatial_dim':spatial_dim,
                 'nodes':model.nodes,
                 'node_coords':np.array([node.coord for node in model.nodes], dtype=float),
                 'node_radiuses':1 / (nodes_num - 1) * radius_ratio * np.ones(len(model.nodes)),
                 'load_map':model.load_map,
                 'kernel_order':kernel_order
                 }
    
    for name in 'volume neumann dirichlet'.split():
        qus = getattr(model, name + '_quadrature_units', None)
        common_data[name + '_quadrature_points'] = iter_quadrature_units(qus)

    for s in search_setup_mixins(processor):
        s.setup(**common_data)
    
    # ss = gen_setup_status(processor)
    
    processor.process()

    matrix = common_data['matrix']
    vector = common_data['vector']
    
    res = np.linalg.solve(matrix, vector)
    
    common_data['node_values'] = res
    
    pp = injector.get(SimpPostProcessor)
    
    for s in search_setup_mixins(pp):
        s.setup(**common_data)
    
    pp.shape_func.partial_order = 1
    
    yss = []
    xs = np.linspace(0, 1).reshape((-1, 1))
    for x in xs:
        yss.append(pp.process(x, None))
    
    yss = np.array(yss)
    
    exp_func = p1d.cases[case]['exp']
    exps = np.array([exp_func(x) for x in xs])
    
    diffs = exps - yss
    
    err = np.abs(diffs[:,0]) / np.abs(exps[:,0]).max(axis=0)
    err = err.max(axis=0)
    assert(err < error_lim)
    
