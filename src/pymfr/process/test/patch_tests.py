'''

@author: "epsilonyuan@gmail.com"
'''
from math import pi, sin, cos
import numpy as np
from pymfr.model.raw_model import MFNode, OneDModel
from injector import Injector
from pymfr.shapefunc.injections import MLSRKShapeFunctionModule
from pymfr.model.injections import OneDSupportNodesSearcherModule, TwoDVisibleSupportNodeSearcherModule
from pymfr.process.injections import get_simp_poission_processor_modules, SimpPostProcessorModule
from pymfr.process.process import SimpProcessor
from pymfr.process.quadrature import iter_quadrature_units, BilinearQuadrangleQuadratureUnit, SegmentQuadratureUnit
from pymfr.misc.tools import search_setup_mixins, gen_setup_status
from pymfr.process.post import SimpPostProcessor
import itertools
from pymfr.model.geom import create_linked_segments_by_nodes, SimpRangle
from nose.tools import ok_

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
    
    def __init__(self, case_index=2, nodes=None):
        self.nodes = nodes
        self.case_index = case_index
    
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
        case_data = self.cases[self.case_index]
        patch_test_data = OneDModel()
        patch_test_data.nodes = nodes
        
        left_dirichlet_node = nodes[0]
        right_dirichlet_node = nodes[-1]
        patch_test_data.dirichlet_nodes = [left_dirichlet_node, right_dirichlet_node]
        patch_test_data.add_dirichlet_load(left_dirichlet_node, case_data['dirichlet_left'])
        patch_test_data.add_dirichlet_load(right_dirichlet_node, case_data['dirichlet_right'])
        
        patch_test_data.set_volume_load(case_data['volume_load'])
        
        patch_test_data.gen_quadrature_units()
        return patch_test_data
        
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
                'error_lim':4e-3  # max_abs_diff/max_abs_exp
                }

    datas = []
    for i in range(4):
        data = {}
        data.update(basic_data)
        data['case_index'] = i
        data['quadrature_degree'] = i + 1
        data['kernel_order'] = i + 1
        datas.append(data)
    
    for data in datas:
        _test_poisson_1d(**data)
        


def _test_poisson_1d(case_index, error_lim, kernel_order, quadrature_degree, nodes_num, radius_ratio, value_dim, spatial_dim, lagrangle_nodes_size):
    vector_size = (nodes_num + lagrangle_nodes_size) * value_dim
    
    
    injector = get_oned_poisson_patch_injector()
    injector.binder.install(SimpPostProcessorModule)
    processor = injector.get(SimpProcessor)
    print(processor)
    p1d = PoissonOneD()
    p1d.case_index = case_index
    p1d.set_nodes_by_num(nodes_num)
    patch_test_data = p1d.gen_model()
    patch_test_data.volume_quadrature_degree = quadrature_degree
    patch_test_data.gen_quadrature_units()
    

    common_data = {'vector':np.zeros(vector_size, dtype=float),
                 'matrix':np.zeros((vector_size, vector_size), dtype=float),
                 'lagrangle_nodes_size':lagrangle_nodes_size,
                 'value_dim':value_dim,
                 'spatial_dim':spatial_dim,
                 'nodes':patch_test_data.nodes,
                 'node_coords':np.array([node.coord for node in patch_test_data.nodes], dtype=float),
                 'node_radiuses':1 / (nodes_num - 1) * radius_ratio * np.ones(len(patch_test_data.nodes)),
                 'load_map':patch_test_data.load_map,
                 'kernel_order':kernel_order
                 }
    
    for name in 'volume neumann dirichlet'.split():
        qus = getattr(patch_test_data, name + '_quadrature_units', None)
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
    
    exp_func = p1d.cases[case_index]['exp']
    exps = np.array([exp_func(x) for x in xs])
    
    diffs = exps - yss
    
    err = np.abs(diffs[:, 0]) / np.abs(exps[:, 0]).max(axis=0)
    err = err.max(axis=0)
    assert(err < error_lim)

def _segment_unit_out(seg):
    vec = seg.end.coord - seg.start.coord
    vec = vec / np.linalg.norm(vec)
    return np.array((vec[1], -vec[0]), dtype=float)

def poisson_linear_2d_exp(coord):
    x = coord[0]
    y = coord[1]
    return np.array((x + 2 * y, 1, 2, 0, 0), dtype=float)
    
def poisson_quadric_2d_exp(coord):
    x = coord[0]
    y = coord[1]
    u = 0.1 * x + 0.3 * y + 0.8 * x ** 2 + 1.2 * x * y + 0.6 * y ** 2
    dx = 0.1 + 1.6 * x + 1.2 * y
    dy = 0.3 + 1.2 * x + 1.2 * y
    ddx = 1.6
    ddy = 1.2
    return np.array((u, dx, dy, ddx, ddy), dtype=float)

def poisson_volume_load_core(exp_func):
    def v(coord, bnd):
        exp = exp_func(coord)
        return ((-exp[3] - exp[4]).reshape(-1), None)
    return v

def poisson_neumann_load_core(exp_func):
    def v(coord, bnd):
        exp = exp_func(coord)
        return (exp[1:3].dot(_segment_unit_out(bnd)).reshape(-1), None)
    
    return v

def poisson_dirichlet_load_core(exp_func):
    def v(coord, bnd):
        exp = exp_func(coord)
        return (exp[0].reshape(-1), (True,))
    
    return v

def gen_bnds_by_node_grids(nodes_grid):
    nodes_grid = np.array(nodes_grid)
    
    bnd_nodes = []
    bnd_nodes.extend(nodes_grid[0])
    bnd_nodes.extend(nodes_grid[1:, -1])
    bnd_nodes.extend(nodes_grid[-1, -2::-1])
    bnd_nodes.extend(nodes_grid[-2:0:-1, 0])
    
    return create_linked_segments_by_nodes(bnd_nodes)

def gen_regular_nodes_segs(xs, ys):
    nodes_grid = []
    
    for  y in ys:
        row = []
        nodes_grid.append(row)
        for  x in xs:
            node = MFNode(np.array((x, y), dtype=float))
            row.append(node)
    
    segs = gen_bnds_by_node_grids(nodes_grid)
    
    nodes = [node for row in nodes_grid for node in row]
    return nodes, segs

def gen_regular_quadrangle_quadrature_units(xs, ys, quadrangle_cls=None):
    if quadrangle_cls is None:
        quadrangle_cls = SimpRangle
    coordss = []
    for y in ys:
        row = []
        coordss.append(row)
        for x in xs:
            row.append(np.array((x, y), dtype=float))
    ret = []
    for i in range(len(ys) - 1):
        for j in range(len(xs) - 1):
            coords = [coordss[i][j], coordss[i][j + 1], coordss[i + 1][j + 1], coordss[i + 1][j]]
            quadrangle = quadrangle_cls(coords)
            ret.append(BilinearQuadrangleQuadratureUnit(quadrangle=quadrangle))
    
    return ret
    

class PatchTestData:
    pass

class PatchTest2D:


    def __init__(self, value_dim, case_index=1, segments=None, nodes=None,
                 volume_quadrature_units=None,
                 neumann_quadrature_units=None,
                 dirichlet_quadrature_units=None,
                                  ):
        
        self.case_index = case_index 
        
        self.left_down = np.array((-1.0, -1.0))
        self.right_up = np.array((1.0, 1.0))
        
        self.dirichlet_left_down = np.array((-1.0001, -1.0001))
        self.dirichlet_right_up = np.array((1.001, -0.999))
        
        self.nodes = nodes
        self.segments = segments
        
        self.volume_quadrature_units = volume_quadrature_units
        self.neumann_quadrature_units = neumann_quadrature_units
        self.dirichlet_quadrature_units = dirichlet_quadrature_units
        
        self.value_dim = value_dim
        self.spatial_dim = 2

    def gen_project_data(self):
        ret = PatchTestData()
        
        case_data = self.cases[self.case_index]
        exp_func = case_data['exp_func']
        order = case_data['order']
        ret.order = order
        load_map = self.gen_load_map(exp_func)
        ret.load_map = load_map
        
        for qu in self.volume_quadrature_units:
            qu.load_key = 'volume'
            qu.degree = order
        
        ret.volume_quadrature_units = self.volume_quadrature_units
        
        def f(seg):
            start = seg.start.coord
            end = seg.end.coord
            upper = self.dirichlet_right_up
            lower = self.dirichlet_left_down
            return ((start <= upper).all() and (start >= lower).all() and 
                    (end <= upper).all() and (end >= lower).all())
                 
        ret.neumann_quadrature_units = [SegmentQuadratureUnit(load_key='neumann', segment=seg, degree=order)
                                     for seg in self.segments if not f(seg)]
        ret.dirichlet_quadrature_units = [SegmentQuadratureUnit(load_key='dirichlet', segment=seg, degree=order)
                                     for seg in self.segments if f(seg)]
        
        ret.nodes = self.nodes
        ret.dirichlet_nodes = self.get_dirichlet_nodes()
        ret.boundaries = self.segments
        ret.value_dim = self.value_dim
        ret.spatial_dim = self.spatial_dim
        
        return ret
    
    def get_dirichlet_nodes(self):
        lower = self.dirichlet_left_down
        upper = self.dirichlet_right_up
        def f(node):
            coord = node.coord
            return ((coord <= upper).all() and (coord >= lower).all())
        return [node for node in self.nodes if f(node)]

class Poisson2D(PatchTest2D):
    cases = [{'exp_func':poisson_linear_2d_exp,
            'order':1
            },
           {'exp_func':poisson_quadric_2d_exp,
            'order':2
            }
           ]
    
    def __init__(self):
        super().__init__(1)
        
    def gen_load_map(self, exp_func):
        load_map = {'volume':poisson_volume_load_core(exp_func),
                    'neumann':poisson_neumann_load_core(exp_func),
                    'dirichlet':poisson_dirichlet_load_core(exp_func)}
        return load_map

def get_twod_poisson_patch_injector():
    injector = Injector()
    modules = [MLSRKShapeFunctionModule, TwoDVisibleSupportNodeSearcherModule]
    modules.extend(get_simp_poission_processor_modules(spatial_dim=2))
    for mod in modules:
        injector.binder.install(mod)
    return injector

def test_twod_poisson():
    for case_index in range(len(Poisson2D.cases)):
        _test_twod_poisson(case_index)

def _test_twod_poisson(case_index, error_lim=2e-2, nodes_num_per_dim=11, node_radius=0.4):
 
    injector = get_twod_poisson_patch_injector()
    injector.binder.install(SimpPostProcessorModule)
    processor = injector.get(SimpProcessor)
    p2d = Poisson2D()
    p2d.case_index = case_index
    xs = np.linspace(-1, 1, nodes_num_per_dim)
    ys = np.linspace(-1, 1, nodes_num_per_dim)
    nodes, segments = gen_regular_nodes_segs(xs, ys)
    p2d.nodes = nodes
    p2d.segments = segments
    
    p2d.volume_quadrature_units = gen_regular_quadrangle_quadrature_units(xs, ys)
    
    ptd = p2d.gen_project_data()
    spatial_dim = ptd.spatial_dim
    value_dim = ptd.value_dim
    
    nodes = ptd.nodes
    
    for i, node in enumerate(nodes):
        node.index = i
    
    lagrangle_nodes = ptd.dirichlet_nodes
    for li, ln in enumerate(lagrangle_nodes, len(nodes)):
        ln.lagrangle_index = li
    
    lagrangle_nodes_size = len(lagrangle_nodes)
    
    node_coords = np.empty((len(nodes) + lagrangle_nodes_size, spatial_dim), dtype=float)
    for node in nodes:
        node_coords[node.index] = node.coord
    
    for ln in lagrangle_nodes:
        node_coords[ln.lagrangle_index] = ln.coord
    
    nodes_num = len(nodes)
    vector_size = (nodes_num + lagrangle_nodes_size) * value_dim
    
    common_data = {}
    common_data.update(ptd.__dict__)
    
    for name in 'volume neumann dirichlet'.split():
        qus = common_data[name + '_quadrature_units']
        common_data[name + '_quadrature_points'] = iter_quadrature_units(qus)
    
    common_data.update({'vector':np.zeros(vector_size, dtype=float),
                 'matrix':np.zeros((vector_size, vector_size), dtype=float),
                 'lagrangle_nodes_size':lagrangle_nodes_size,
                 'node_coords':node_coords,
                 'node_radiuses':node_radius * np.ones(len(ptd.nodes)),
                 'kernel_order':ptd.order,
                 'segments':ptd.boundaries
                 })
    
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
    
    yss = []
    xs = list(itertools.product(np.linspace(-1, 1), np.linspace(-1, 1)))
    for x in xs:
        yss.append(pp.process(x, None))
    
    yss = np.array(yss)
    
    exp_func = p2d.cases[p2d.case_index]['exp_func']
    exps = np.array([exp_func(x) for x in xs], dtype=float)
    
    diff = np.abs(yss[:, 0] - exps[:, 0]).max()
    error = diff / np.abs(exps[:, 0]).max()
    
    ok_(error < error_lim)

