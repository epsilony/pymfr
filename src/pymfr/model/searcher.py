'''

@author: "epsilonyuan@gmail.com"
'''

import numpy as np
from scipy.spatial import cKDTree as KDTree
from pymfr.misc.tools import ensure_sequence
from numpy.linalg import norm
from numbers import Number
from collections import Callable
from math import ceil, sqrt
import sys
import itertools
from abc import ABCMeta, abstractmethod
from pymfr.misc.mixin import SetupMixin
    
def _max_len(segments):
    return max(norm(segment.end.coord - segment.start.coord) for segment in segments)

def norm_2d(vec2d):
    return sqrt(vec2d[0] * vec2d[0] + vec2d[1] * vec2d[1])

def distance_square_2d(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

def _distance_to_seg(x, seg):
    start = seg.start.coord
    end = seg.end.coord
    u = end - start
    v = x - start
    norm_u = norm_2d(u)
    if 0 == norm_u:
        raise ValueError('zero length segment:%s' + str(seg))
    t = np.dot(v, u) / norm_u
    if t <= 0:
        return norm_2d(start - x)
    elif t >= norm_u:
        return norm_2d(v)
    else:
        return norm_2d(u * t / norm_u + start - x)


class NodeSearcher(metaclass=ABCMeta):
    @abstractmethod
    def search_indes(self, x, rad):
        pass

class SegmentSearcher(metaclass=ABCMeta):
    @abstractmethod
    def search_indes(self, x, rad):
        pass

class RawNodeSearcher(NodeSearcher, SetupMixin):
    
    __prerequisites__ = ['nodes']
        
    def search(self, x, rad, eps=0):
        r = rad * (1 + eps)
        return [node for node in self.nodes if norm(node.coord - x) < r]
    
    def search_indes(self, x, rad, eps=0):
        nodes = self.nodes
        r = rad * (1 + eps)
        return [i for i in range(len(nodes)) if norm(nodes[i].coord - x) < r]


class RawSegmentSearcher(SegmentSearcher, SetupMixin):        

    __prerequisites__ = ['segments']
            
    def search_indes(self, x, rad, eps=0):
        if eps < 0:
            raise ValueError()
        r = rad * (1 + eps)
        segments = self.segments
        return [i for i in range(len(segments)) if _distance_to_seg(x, segments[i]) < r]
    
    def search(self, x, rad, eps=0):
        if eps < 0:
            raise ValueError()
        rad *= (1 + eps)
        segments = self.segments
        return [segment for segment in segments if _distance_to_seg(x, segment) < rad]

class KDTreeNodeSearcher(NodeSearcher, SetupMixin):
    
    __prerequisites__ = ['nodes']
    
    @property
    def nodes(self):
        return self._nodes
    
    @nodes.setter
    def nodes(self, nodes):
        self._nodes = nodes
        coords = [node.coord for node in nodes]
        self.kd_tree = KDTree(coords)
    
    def search_indes(self, x, rad, eps=0):
        rad *= (1 + eps)
        return self.kd_tree.query_ball_point(x, rad)
    
    def search(self, x, rad, eps=0):
        indes = self.search_indes(x, rad, eps)
        nodes = self.nodes
        return [nodes[i] for i in indes]

def _estimate_loosen(seg_sizes):
        
        mean = np.mean(seg_sizes)
        std = np.mean(seg_sizes)
        exp_loosen = (mean + std) / 2
        return exp_loosen

class KDTreeSegmentSearcher(SegmentSearcher, SetupMixin):
    
    __prerequisites__ = ['segments']
    __optionals__ = [('loosen', None)]
    __after_setup__ = ['build_tree']
    
    def build_tree(self):
        segments = self.segments
        loosen = self.loosen
        seg_sizes = [norm(seg.end.coord - seg.start.coord) for seg in segments]
        if loosen is None:
            self.loosen = _estimate_loosen(seg_sizes)
        elif isinstance(loosen, Number):
            if loosen <= 0:
                raise ValueError()
            self.loosen = float(loosen)
        elif isinstance(loosen, Callable):
            self.loosen = loosen(segments)
        else:
            raise ValueError()
        
        self._gen_kdtree(seg_sizes)
    
    def _gen_kdtree(self, seg_sizes):
        loosen = self.loosen
        segments = self.segments
        keys = []
        medium_indes = []
        rad_plus = 0
        for i in range(len(segments)):
            segment = segments[i]
            seg_size = seg_sizes[i]
            if seg_size < loosen * 2:
                keys.append((segment.start.coord + segment.end.coord) / 2)
                medium_indes.append(i)
                half_seg_size = seg_size / 2
                if half_seg_size > rad_plus:
                    rad_plus = half_seg_size
            else:
                new_keys_num = ceil(seg_size / loosen / 2)
                start = 1 / new_keys_num / 2
                stop = 1 - start
                ts = np.linspace(start, stop, new_keys_num)
                u = segment.end.coord - segment.start.coord
                half_fake_seg_size = seg_size / new_keys_num / 2
                if half_fake_seg_size > rad_plus:
                    rad_plus = half_fake_seg_size
                for t in ts:
                    keys.append((segment.start.coord + u * t))
                    medium_indes.append(i)
        self.kd_tree = KDTree(keys)
        self.rad_plus = rad_plus
        
        if len(keys) == len(segments):
            self.medium_indes = None
        else:
            self.medium_indes = np.array(medium_indes, dtype=int)
            self._indes_generation = np.zeros(len(self.segments), dtype=int)
            self._current_generation = 1
            
    def rough_search_indes(self, x, rad, eps=0):
        r = (rad + self.rad_plus) * (1 + eps)
        indes = self.kd_tree.query_ball_point(x, r, eps=eps)
        indes_generation = self._indes_generation
        if self.medium_indes is not None:
            self._shift_current_generation()
            medium_indes = self.medium_indes
            current_generation = self._current_generation
            result = []
            for i in indes:
                index = medium_indes[i]
                if indes_generation[index] == current_generation:
                    continue
                else:
                    indes_generation[index] = current_generation
                    result.append(index)
            return result
        else:
            return indes
    
    def _shift_current_generation(self):
        if self._current_generation == sys.maxsize:
            self._current_generation = 1
            self.index_generation.fill(0)
        else:
            self._current_generation += 1
    
    def search_indes(self, x, rad, eps=0):
        rough_indes = self.rough_search_indes(x, rad, eps)
        segments = self.segments
        r = rad * (1 + eps)
        return [i for i in rough_indes if _distance_to_seg(x, segments[i]) < r]
    
    def search(self, x, rad, eps=0):
        rough_indes = self.rough_search_indes(x, rad, eps)
        segments = self.segments
        r = rad * (1 + eps)
        return [segments[i] for i in rough_indes if _distance_to_seg(x, segments[i]) < r]

class SupportNodeSearcher(metaclass=ABCMeta):
    @abstractmethod
    def search_indes(self, x):
        pass

class RawSupportNodeSearcher(SupportNodeSearcher, SetupMixin):
    
    __prerequisites__ = ['nodes']
    __optionals__ = [('node_radiuses', None)]
    __after_setup__ = ['build']
    
    
    def build(self):
        if self.node_radiuses is None:
            self.node_radiuses = [node.radius for node in self.nodes]
        self.node_coords = [node.coord for node in self.nodes]
        return self
        
    def search_indes(self, x, eps=0):
        return [i for i in range(len(self.nodes)) if norm(self.nodes[i].coord - x) < 
                (self.node_radiuses[i] if self.node_radiuses is not None else self.nodes[i].radius) * (1 + eps)]
    
    def search_nodes(self, x, eps=0):
        return [self.nodes[i] for i in range(len(self.nodes)) if norm(self.nodes[i].coord - x) < 
                (self.node_radiuses[i] if self.node_radiuses is not None else self.nodes[i].radius) * (1 + eps)]

class KDTreeSupportNodeSearcher(SupportNodeSearcher, SetupMixin):
    
    __prerequisites__ = ['nodes']
    __optionals__ = [('node_radiuses', None), ('loosen', None)]
    __after_setup__ = ['build_tree']
    
    def build_tree(self):
        nodes = self.nodes
        if self.node_radiuses is None:
            rads = [node.radius for node in nodes]
            self.node_radiuses = rads
        
        loosen = self.loosen
        if loosen is None:
            self.loosen = self.estimate_loosen(self.node_radiuses)
        elif isinstance(loosen, Number):
            self.loosen = loosen
        else:
            self.loosen = loosen(nodes, self.node_radiuses)
        
        self.node_coords = np.array([node.coord for node in nodes], dtype=float)
        self._build_kd_tree(self.node_radiuses)
        return self
    
    def estimate_loosen(self, rads):
        mean = np.mean(rads)
        std = np.std(rads)
        return mean + std
    
    def _build_kd_tree(self, rads):
        keys = []
        medium_indes = []
        loosen = self.loosen
        coords = self.node_coords
        for i in range(len(coords)):
            coord = coords[i]
            rad = rads[i]
            if rad <= loosen:
                keys.append(coord)
                medium_indes.append(i)
            else:
                alter_num_per_dim = ceil(rad / loosen)
                for alter_coord in itertools.product(*[np.linspace(c - rad + loosen, c + rad - loosen, alter_num_per_dim) for c in coord]):
                    keys.append(alter_coord)
                    medium_indes.append(i)
        if len(coords) < len(keys):
            self.medium_indes = np.array(medium_indes, dtype=int)
            self._indes_generation = np.zeros(len(self.nodes), dtype=int)
            self._current_generation = 1
        self.kd_tree = KDTree(keys)
                
    def search_indes(self, x):
        r = self.loosen
        indes = self.kd_tree.query_ball_point(x, r, p=float('inf'))
        indes_generation = self._indes_generation
        if self.medium_indes is not None:
            self._shift_current_generation()
            medium_indes = self.medium_indes
            current_generation = self._current_generation
            result = []
            for i in indes:
                index = medium_indes[i]
                if indes_generation[index] == current_generation:
                    continue
                
                indes_generation[index] = current_generation
                rad = self.node_radiuses[index]
                coord = self.node_coords[index]
                if np.linalg.norm(x - coord) >= rad:
                    continue
                result.append(index)
            return result
        else:
            return [i for i in indes if np.linalg.norm(self.node_coords[i] - x) < self.node_radiuses[i]]
        
    def search_nodes(self, x):
        return [self.nodes[i] for i in self.search_indes(x)]
    
    def _shift_current_generation(self):
        if self._current_generation == sys.maxsize:
            self._current_generation = 1
            self.index_generation.fill(0)
        else:
            self._current_generation += 1   
            
class _SupportDomainNodeSegmentSearcher2D:
    def __init__(self, support_node_searcher, segment_searcher, **kwargs):
        self.support_node_searcher = support_node_searcher
        self.segment_searcher = segment_searcher
    
    def search_node_segment_indes(self, x):
        node_indes = self.search_node_indes(x)
        sns = self.support_node_searcher
        farthest_dist = max(distance_square_2d(sns.node_coords[i], x) for i in node_indes)
        farthest_dist = farthest_dist ** 0.5
        segment_indes = self.segment_searcher.search_indes(x, farthest_dist)
        
        return node_indes, segment_indes
    
    def search_node_indes(self, x):
        sns = self.support_node_searcher
        node_indes = sns.search_indes(x)
        return node_indes
    
class VisibleSupportNodeSearcher2D(SupportNodeSearcher, SetupMixin):
    
    __optionals__ = [('perturb_distance', 1e-6)]
    
    def __init__(self,
              support_node_searcher,
              segment_searcher,
              perturb_distance=None
              ):
        
        self.setup(perturb_distance=perturb_distance)
        self.support_node_searcher = _SupportDomainNodeSegmentSearcher2D(support_node_searcher, segment_searcher)
        self.support_node_searhcer = support_node_searcher
        self.segment_searcher = segment_searcher

    def search_indes(self, x, bnd=None):
        node_indes, segment_indes = self.support_node_searcher.search_node_segment_indes(x)
        if len(segment_indes) == 0:
            return node_indes
        else:
            return self.visible_filter(x, bnd, node_indes, segment_indes)
        
    def visible_filter(self, x, bnd, node_indes, segment_indes):
        pb_x = x if bnd is None else self._perturb_x(x, bnd) 
        all_segs = self.segment_searcher.segments
        segs = [] if bnd is None else [bnd]
        for i in segment_indes:
            seg = all_segs[i]
            if seg is bnd:
                continue
            if np.cross(seg.end.coord - seg.start.coord, pb_x - seg.start.coord) > 0:
                segs.append(seg)
        
        all_nodes = self.support_node_searhcer.nodes
        
        
        filtered_mask = np.zeros(len(node_indes), dtype=np.byte)
        for seg in segs:
            for i in range(len(node_indes)):
                if filtered_mask[i]:
                    continue
                index = node_indes[i]
                node = all_nodes[index]
                if seg.start is node or seg.end is node:
                    continue
                coord = node.coord
                u = seg.end.coord - seg.start.coord
                if np.cross(u, coord - seg.start.coord) >= 0:
                    continue
                if seg is bnd:
                    filtered_mask[i] = 1
                    continue
                v = coord - pb_x
                if np.cross(seg.start.coord - pb_x, v) < 0 or np.cross(seg.end.coord - pb_x, v) > 0:
                    continue
                filtered_mask[i] = 1
        
        node_indes = [node_indes[i] for i in range(len(node_indes)) if filtered_mask[i] == 0]
        return node_indes
        
    _COS_LIMIT = 1 / 2 ** 0.5
    def _perturb_x(self, x, bnd):
        s = bnd.start.coord
        e = bnd.end.coord
        n = bnd.succ.end.coord - e
        n = n / norm_2d(n)
        p = bnd.pred.start.coord - s
        p = p / norm_2d(p)
        v = e - s
        bnd_length = norm_2d(v)
        v = v / bnd_length
        
        
        COS_LIMIT = self._COS_LIMIT
        perturb_distance = self.perturb_distance
        vp = (v + p) / 2
        norm_vp = norm_2d(vp)
        if norm_vp == 0:
            left_limit = 0
        else:
            vp = vp / norm_vp
            cos_vp_v = np.dot(vp, v)
            if cos_vp_v < COS_LIMIT:
                left_limit = 0
            else:
                sin_vp_v = np.cross(v, vp)
                left_limit = perturb_distance / sin_vp_v * cos_vp_v
        
        vn = (-v + n) / 2
        norm_vn = norm_2d(vn)
        if norm_vn == 0:
            right_limit = 0
        else:
            vn = vn / norm_vn
            
            cos_vn_v = np.dot(vn, -v)
            
    
            if cos_vn_v < COS_LIMIT:
                right_limit = 0
            else:
                sin_vn_v = np.cross(vn, -v)
                right_limit = perturb_distance / sin_vn_v * cos_vn_v
        
        if left_limit > bnd_length - right_limit:
            raise ValueError('the angle around %s is too small: ' % (str(bnd)))
        
        xs = x - s
        xs_len = norm_2d(xs)
        
        if xs_len < left_limit:
            perturb_ori = s + v * left_limit
        elif xs_len > bnd_length - right_limit:
            perturb_ori = s + v * (bnd_length - right_limit)
        else:
            perturb_ori = x
        
        perturb_v = np.array((-v[1], v[0]), dtype=float)
        return perturb_ori + perturb_v * perturb_distance
