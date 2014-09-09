'''

@author: "epsilonyuan@gmail.com"
'''

import numpy as np
from pymfr.process.quadrature import SegmentQuadratureUnit, SymmetricTriangleQuadratureUnit, \
    BilinearQuadrangleQuadratureUnit
from nose.tools import eq_, ok_, assert_almost_equal
import sympy

class MockNode:
    def __init__(self, coord):
        self.coord = coord

class MockSegment:
    def __init__(self, startcoord, endcoord):
        self.start = MockNode(startcoord)
        self.end = MockNode(endcoord)

def test_segment_quadrature_unit():
    segment = MockSegment(np.array([0.1, -0.2], dtype=float), np.array([3, 2.1], dtype=float))
    squ = SegmentQuadratureUnit(None, None, segment)
    squ.degree = 1
    eq_(squ.legendre_degree, 1)
    
    u = segment.end.coord - segment.start.coord
    seg_length = np.linalg.norm(u)
    sub_units = squ.gen_subunits()
    
    act = 0
    for su in sub_units:
        ok_(su.father is squ)
        act += su.weight
    
    assert_almost_equal(act, seg_length)
    
    x, y, t = sympy.symbols('x,y,t')
    
    f = 2 - x + y + 0.2 * x ** 2 + 0.3 * x * y - 0.4 * y ** 2 + 0.6 * x ** 3 - 0.7 * x ** 2 * y - 0.1 * x * y ** 2 - 0.4 * y ** 3
    
    start = segment.start.coord
    end = segment.end.coord
    ft = f.subs(x, (1 - t) * start[0] + t * end[0]).subs(y, (1 - t) * start[1] + t * end[1])
    fti = (ft * seg_length).integrate(t)
    
    exp = fti.subs(t, 1) - fti.subs(t, 0)
    act = 0
    squ.degree = 3
    for su in squ.gen_subunits():
        act += su.weight * f.subs(x, su.coord[0]).subs(y, su.coord[1])
    assert_almost_equal(exp, act)

class Mock:
    pass
    

def _int_over_polygon(poly_coords, x, y, f):
    exp = 0
    for i in range(-1, len(poly_coords) - 1):
        start = poly_coords[i]
        u = poly_coords[i + 1] - poly_coords[i]
        fiy = f.integrate(y)
        fiy = fiy.subs(y, (x - start[0]) / u[0] * u[1] + start[1]) - fiy.subs(y, 0)
        fixy = fiy.integrate(x)
        exp -= fixy.subs(x, poly_coords[i + 1][0]) - fixy.subs(x, start[0])
    
    return exp

def test_sym_triangle_quadrature():
    triangle = Mock()
    triangle.coords = np.array([
                              [0.1, 0.2],
                              [1.1, 0.5],
                              [-1, 1.1]], dtype=float)
    
    tri_coords = triangle.coords
    tri_area = np.cross((tri_coords[0] - tri_coords[1]), (tri_coords[2] - tri_coords[1]) / 2)
    tri_area = abs(tri_area)
    stq = SymmetricTriangleQuadratureUnit(None, None, triangle)
    
    stq.degree = 1
    
    act = 0
    for qp in stq.gen_subunits():
        act += qp.weight
    assert_almost_equal(tri_area, act)
    
    x, y = sympy.symbols('x,y')
    f = 2 - x + y + 0.2 * x ** 2 + 0.3 * x * y - 0.4 * y ** 2 + 0.6 * x ** 3 - 0.7 * x ** 2 * y - 0.1 * x * y ** 2 - 0.4 * y ** 3
    exp = _int_over_polygon(tri_coords, x, y, f)
    
    stq.degree = 3
    act = 0
    for qp in stq.gen_subunits():
        act += qp.weight * f.subs(x, qp.coord[0]).subs(y, qp.coord[1])
    assert_almost_equal(exp, act)
    
def test_quadrangle_quadrature_unit():
    quadrangle = Mock()
    quadrangle.coords = np.array([
                              [0.1, 0.2],
                              [1.1, 0.5],
                              [1.2, 1.3],
                              [-1, 1.1]], dtype=float)
    
    coords = quadrangle.coords
    qqu = BilinearQuadrangleQuadratureUnit(None, None, quadrangle)
    x, y = sympy.symbols('x,y')
    f_1 = 1 + 0 * x
    f_2 = 2 - x + y + 0.2 * x ** 2 + 0.3 * x * y - 0.4 * y ** 2 + 0.6 * x ** 3 - 0.7 * x ** 2 * y - 0.1 * x * y ** 2 - 0.4 * y ** 3
    
    degrees = 1, 3
    fs = f_1, f_2
    
    for f, degree in zip(fs, degrees):
        exp = _int_over_polygon(coords, x, y, f)
        
        qqu.degree = degree
        act = 0
        for qp in qqu.gen_subunits():
            act += qp.weight * f.subs(x, qp.coord[0]).subs(y, qp.coord[1])
        assert_almost_equal(exp, act)
