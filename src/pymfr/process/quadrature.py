'''
@author: "epsilonyuan@gmail.com"
'''

import numpy as np
from math import ceil


class QuadratureUnit:
    
    def __init__(self, father=None, load_key=None):
        self.father = father
        self._load_key = load_key
        
    @property
    def load_key(self):
        if self._load_key is None:
            if self.father is None:
                return None
            return self.father.load_key
        return self._load_key
    
    @load_key.setter
    def load_key(self, value):
        self._load_key = value
    
    def gen_subunits(self):
        return None
    
class QuadraturePoint(QuadratureUnit):
    
    def __init__(self, father=None, load_key=None, weight=0, coord=None, bnd=None):
        super().__init__(father, load_key)
        self.weight = weight
        self.coord = coord
        self.bnd = bnd

class _GaussianQuadratureData:
    _PTS_WEIGHTS_LIST = [np.polynomial.legendre.leggauss(deg) for deg in range(1, 11)]

class SegmentQuadratureUnit(QuadratureUnit):
    
    
    
    def __init__(self, father=None, load_key=None, segment=None, degree=2):
        super().__init__(father, load_key)
        self.segment = segment
        self.degree = degree
    
    def gen_subunits(self):
        legendre_degree = self.legendre_degree
        
        pts, weights = _GaussianQuadratureData._PTS_WEIGHTS_LIST[legendre_degree - 1]
        segment = self.segment
        start_coord = segment.start.coord
        u = segment.end.coord - start_coord
        segment_length = np.linalg.norm(u)
        
        weight_factor = segment_length / 2
        result = []
        for pt, weight in zip(pts, weights):
            coord = u * (pt + 1) / 2 + start_coord
            qp_weight = weight * weight_factor
            result.append(QuadraturePoint(self, None, qp_weight, coord, self.segment))
        
        return result
    
    @property
    def degree(self):
        return int(ceil(self.legendre_degree * 2 - 1))
        
    @degree.setter
    def degree(self, alg_degree):
        legendre_degree = int(ceil((alg_degree + 1) / 2))
        if legendre_degree < 1 or legendre_degree > 10:
            raise ValueError()
        self.legendre_degree = legendre_degree

class IntervalQuadratureUnit(QuadratureUnit):
    
    def __init__(self, father=None, load_key=None, interval=None, degree=2):
        super().__init__(father, load_key)
        self.interval = interval
        self.degree = degree
    
    def gen_subunits(self):
        legendre_degree = self.legendre_degree
        
        pts, weights = _GaussianQuadratureData._PTS_WEIGHTS_LIST[legendre_degree - 1]
        interval = self.interval
        start_coord = interval[0]
        u = interval[1] - start_coord
        interval_length = abs(interval[1] - interval[0])
        
        weight_factor = interval_length / 2
        result = []
        for pt, weight in zip(pts, weights):
            coord = u * (pt + 1) / 2 + start_coord
            qp_weight = weight * weight_factor
            result.append(QuadraturePoint(self, None, qp_weight, coord, None))
        
        return result
    
    @property
    def degree(self):
        return int(ceil(self.legendre_degree * 2 - 1))
        
    @degree.setter
    def degree(self, alg_degree):
        legendre_degree = int(ceil((alg_degree + 1) / 2))
        if legendre_degree < 1 or legendre_degree > 10:
            raise ValueError()
        self.legendre_degree = legendre_degree
        
class SymmetricTriangleQuadratureUnit(QuadratureUnit):
    def __init__(self, father=None, load_key=None, triangle=None, degree=2):
        super().__init__(father, load_key)
        self.triangle = triangle
        self.degree = degree
        
    def gen_subunits(self):
        results = []
        weights = _SymmetricTriangleQuadratureData.WEIGHTSS[self.degree - 1]
        barycentric_coords = _SymmetricTriangleQuadratureData.BARYCENTRIC_COORDSS[self.degree - 1].reshape((-1, 3))
        
        triangle = self.triangle
        tri_coords = triangle.coords
        t_cross = np.cross(tri_coords[1] - tri_coords[0], tri_coords[2] - tri_coords[0])
        
        triangle_area = abs(t_cross) / 2 if np.isscalar(t_cross) else np.linalg.norm(t_cross) / 2
        for weight, bc in zip(weights, barycentric_coords):
            coord = bc.dot(tri_coords)
            results.append(QuadraturePoint(self, None, weight * triangle_area, coord, triangle))
        return results
        
class _SymmetricTriangleQuadratureData:
    MAX_ALGEBRAIC_ACCURACY = 8;
    MIN_ALGEBRAIC_ACCURACY = 1;
    NUM_PTS = [ 1, 3, 4, 6, 7, 12, 13, 16 ];
    WEIGHTSS = [
            [ 1 ],
            [ 1 / 3, 1 / 3, 1 / 3 ],
            [ -27 / 48, 25 / 48, 25 / 48, 25 / 48 ],
            [ 0.109951743655322, 0.109951743655322, 0.109951743655322, 0.223381589678011, 0.223381589678011,
                    0.223381589678011 ],
            [ 0.225000000000000, 0.125939180544827, 0.125939180544827, 0.125939180544827, 0.132394152788506,
                    0.132394152788506, 0.132394152788506 ],
            [ 0.050844906370207, 0.050844906370207, 0.050844906370207, 0.116786275726379, 0.116786275726379,
                    0.116786275726379, 0.082851075618374, 0.082851075618374, 0.082851075618374, 0.082851075618374,
                    0.082851075618374, 0.082851075618374 ],
            [ -0.149570044467682, 0.175615257433208, 0.175615257433208, 0.175615257433208, 0.053347235608838,
                    0.053347235608838, 0.053347235608838, 0.077113760890257, 0.077113760890257, 0.077113760890257,
                    0.077113760890257, 0.077113760890257, 0.077113760890257 ],
            [ 0.144315607677787, 0.095091634267285, 0.095091634267285, 0.095091634267285, 0.103217370534718,
                    0.103217370534718, 0.103217370534718, 0.032458497623198, 0.032458497623198, 0.032458497623198,
                    0.027230314174435, 0.027230314174435, 0.027230314174435, 0.027230314174435, 0.027230314174435,
                    0.027230314174435 ], ];
    WEIGHTSS = [np.array(w, dtype=float) for w in WEIGHTSS]
    BARYCENTRIC_COORDSS = [
            [ 1 / 3, 1 / 3, 1 / 3 ],
            [ 2 / 3, 1 / 6, 1 / 6, 1 / 6, 2 / 3, 1 / 6, 1 / 6, 1 / 6, 2 / 3 ],
            [ 1 / 3, 1 / 3, 1 / 3, 0.6, 0.2, 0.2, 0.2, 0.6, 0.2, 0.2, 0.2, 0.6 ],
            [ 0.816847572980459, 0.091576213509771, 0.091576213509771, 0.091576213509771, 0.816847572980459,
                    0.091576213509771, 0.091576213509771, 0.091576213509771, 0.816847572980459, 0.108103018168070,
                    0.445948490915965, 0.445948490915965, 0.445948490915965, 0.108103018168070, 0.445948490915965,
                    0.445948490915965, 0.445948490915965, 0.108103018168070 ],
            [ 1 / 3, 1 / 3, 1 / 3, 0.797426985353087, 0.101286507323456, 0.101286507323456, 0.101286507323456,
                    0.797426985353087, 0.101286507323456, 0.101286507323456, 0.101286507323456, 0.797426985353087,
                    0.059715871789770, 0.470142064105115, 0.470142064105115, 0.470142064105115, 0.059715871789770,
                    0.470142064105115, 0.470142064105115, 0.470142064105115, 0.059715871789770 ],
            [ 0.873821971016996, 0.063089014491502, 0.063089014491502, 0.063089014491502, 0.873821971016996,
                    0.063089014491502, 0.063089014491502, 0.063089014491502, 0.873821971016996, 0.501426509658179,
                    0.249286745170910, 0.249286745170910, 0.249286745170910, 0.501426509658179, 0.249286745170910,
                    0.249286745170910, 0.249286745170910, 0.501426509658179, 0.636502499121399, 0.310352451033784,
                    0.053145049844817, 0.636502499121399, 0.053145049844817, 0.310352451033784, 0.310352451033784,
                    0.636502499121399, 0.053145049844817, 0.310352451033784, 0.053145049844817, 0.636502499121399,
                    0.053145049844817, 0.636502499121399, 0.310352451033784, 0.053145049844817, 0.310352451033784,
                    0.636502499121399 ],
            [ 1 / 3, 1 / 3, 1 / 3, 0.479308067841920, 0.260345966079040, 0.260345966079040, 0.260345966079040,
                    0.479308067841920, 0.260345966079040, 0.260345966079040, 0.260345966079040, 0.479308067841920,
                    0.869739794195568, 0.065130102902216, 0.065130102902216, 0.065130102902216, 0.869739794195568,
                    0.065130102902216, 0.065130102902216, 0.065130102902216, 0.869739794195568, 0.638444188569810,
                    0.312865496004874, 0.048690315425316, 0.638444188569810, 0.048690315425316, 0.312865496004874,
                    0.312865496004874, 0.638444188569810, 0.048690315425316, 0.312865496004874, 0.048690315425316,
                    0.638444188569810, 0.048690315425316, 0.638444188569810, 0.312865496004874, 0.048690315425316,
                    0.312865496004874, 0.638444188569810 ],
            [ 1 / 3, 1 / 3, 1 / 3, 0.081414823414554, 0.459292588292723, 0.459292588292723, 0.459292588292723,
                    0.081414823414554, 0.459292588292723, 0.459292588292723, 0.459292588292723, 0.081414823414554,
                    0.658861384496480, 0.170569307751760, 0.170569307751760, 0.170569307751760, 0.658861384496480,
                    0.170569307751760, 0.170569307751760, 0.170569307751760, 0.658861384496480, 0.898905543365938,
                    0.050547228317031, 0.050547228317031, 0.050547228317031, 0.898905543365938, 0.050547228317031,
                    0.050547228317031, 0.050547228317031, 0.898905543365938, 0.008394777409958, 0.263112829634638,
                    0.728492392955404, 0.008394777409958, 0.728492392955404, 0.263112829634638, 0.263112829634638,
                    0.008394777409958, 0.728492392955404, 0.263112829634638, 0.728492392955404, 0.008394777409958,
                    0.728492392955404, 0.008394777409958, 0.263112829634638, 0.728492392955404, 0.263112829634638,
                    0.008394777409958 ] ];    
    BARYCENTRIC_COORDSS = [np.array(bc, dtype=float) for bc in BARYCENTRIC_COORDSS]

class BilinearQuadrangleQuadratureUnit(QuadratureUnit):
    _MAT = np.array([[1, -1, -1, 1],
     [1, -1, 1, -1],
     [1, 1, 1, 1],
     [1, 1, -1, -1]
     ], dtype=float)
    
    _MAT_INV = np.linalg.inv(_MAT)
    
    def __init__(self, father=None, load_key=None, quadrangle=None, degree=2):
        super().__init__(father, load_key)
        self.quadrangle = quadrangle
        self.degree = degree
        
    def gen_subunits(self):
        x_coef, y_coef = self._calc_bilinear_map_coefs()
        legendre_degree = int(ceil((self.degree + 1 + 1) / 2))
        
        pts, weights = _GaussianQuadratureData._PTS_WEIGHTS_LIST[legendre_degree - 1]
        results = []
        for i in range(len(weights)):
            for j in range(len(weights)):
                u = pts[i]
                v = pts[j]
                w = weights[i] * weights[j]
                x = x_coef[0] + u * x_coef[1] + v * x_coef[2] + u * v * x_coef[3]
                y = y_coef[0] + u * y_coef[1] + v * y_coef[2] + u * v * y_coef[3]
                coord = np.array([x, y], dtype=float)
                jacob = np.array([
                              [x_coef[1] + x_coef[3] * v, x_coef[2] + x_coef[3] * u],
                              [y_coef[1] + y_coef[3] * v, y_coef[2] + y_coef[3] * u]  
                                ], dtype=float)
                
                results.append(QuadraturePoint(self, None, abs(np.linalg.det(jacob)) * w, coord, self.quadrangle))
        return results
        
    def _calc_bilinear_map_coefs(self):
        coords = np.asarray(self.quadrangle.coords)
        mat_inv = BilinearQuadrangleQuadratureUnit._MAT_INV
        x_coef = mat_inv.dot(coords[:, 0])
        y_coef = mat_inv.dot(coords[:, 1])
        return x_coef, y_coef

        
def iter_quadrature_units(quadrature_units):
    stack = list(quadrature_units)
    while stack:
        qu = stack.pop()
        if isinstance(qu, QuadraturePoint):
            yield qu
        else:
            sub_units = qu.gen_subunits()
            if sub_units is not None:
                stack.extend(sub_units)
