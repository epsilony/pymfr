'''

@author: "epsilonyuan@gmail.com"
'''
from pymfr.process.quadrature import IntervalQuadratureUnit, QuadraturePoint

class MFNode:
    
    def __init__(self, coord=None):
        self.coord = coord

class OneDModel:
    
    def __init__(self, nodes=None, volume_quadrature_degree=2):
        self.nodes = nodes
        self.load_map = {}
        self.dirichlet_nodes = []
        self.neumann_nodes = []
        self.volume_quadrature_degree = volume_quadrature_degree
    
    def gen_quadrature_units(self):
        nodes = self.nodes
        
        degree = self.volume_quadrature_degree
        volume_quadrature_units = []
        for i in range(len(nodes) - 1):
            start = nodes[i]
            end = nodes[i + 1]
            iqu = IntervalQuadratureUnit(None, 'volume', (start.coord, end.coord), degree)
            volume_quadrature_units.append(iqu)
        self.volume_quadrature_units = volume_quadrature_units
        
        self.neumann_quadrature_units = []
        for neumann_node in self.neumann_nodes:
            self.neumann_quadrature_units.append(
                           QuadraturePoint(None, neumann_node, 1, neumann_node.coord, neumann_node))
        
        self.dirichlet_quadrature_units = []
        for dirichlet_node in self.dirichlet_nodes:
            self.dirichlet_quadrature_units.append(
                           QuadraturePoint(None, dirichlet_node, 1, dirichlet_node.coord, dirichlet_node))
    
    def add_neumann_load(self, node, load_core):
        self.load_map[node] = load_core
        self.neumann_node = node
    
    def add_dirichlet_load(self, node, load_core):
        self.load_map[node] = load_core
        self.dirichlet_node = node
    
    def set_volume_load(self, load_core):
        self.load_map['volume'] = load_core
    
