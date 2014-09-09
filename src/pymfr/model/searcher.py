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
    
def _max_len(segments):
    return max(norm(segment.end.coord - segment.start.coord) for segment in segments)

def norm_2d(vec2d):
    return sqrt(vec2d[0] * vec2d[0] + vec2d[1] * vec2d[1])

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

class RawNodeSearcher:
    def __init__(self, nodes):
        self.nodes = ensure_sequence(nodes)
        
    def search(self, x, rad, eps=0):
        r = rad * (1 + eps)
        return [node for node in self.nodes if norm(node.coord - x) < r]
    
    def search_indes(self, x, rad, eps=0):
        nodes = self.nodes
        r = rad * (1 + eps)
        return [i for i in range(len(nodes)) if norm(nodes[i].coord - x) < r]


class RawSegmentSearcher:        
    def __init__(self, segments):
        segments = ensure_sequence(segments)
        self.segments = segments
            
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

class KDTreeNodeSearcher:
    def __init__(self, nodes):
        nodes = ensure_sequence(nodes)
        self.nodes = nodes
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

class KDTreeSegmentSearcher:
    def __init__(self, segments, loosen=None):
        segments = ensure_sequence(segments)
        self.segments = segments
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
            self.medium_indes = np.empty((len(keys), 2), dtype=int)
            self.medium_indes[:, 0] = medium_indes
            self.medium_indes[:, 1] = 0
            self._current_generation = 0
                
    
    def rough_search_indes(self, x, rad, eps=0):
        r = (rad + self.rad_plus) * (1 + eps)
        indes = self.kd_tree.query_ball_point(x, r, eps=eps)
        if self.medium_indes is not None:
            self._shift_current_generation()
            medium_indes = self.medium_indes
            current_generation = self._current_generation
            result = []
            for i in indes:
                index, generation = medium_indes[i]
                if generation == current_generation:
                    continue
                else:
                    medium_indes[i, 1] = current_generation
                    result.append(index)
            return result
        else:
            return indes
    
    def _shift_current_generation(self):
        if self._current_generation == sys.maxsize:
            self._current_generation = 0
            self.medium_indes[:, 1] = -1
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


class RawSupportNodeSearcher:
    
    def __init__(self, nodes, rads=None):
        self.rads = rads
        self.nodes = nodes
        
    def search_indes(self, x, eps=0):
        return [i for i in range(len(self.nodes)) if norm(self.nodes[i].coord - x) < 
                (self.rads[i] if self.rads is not None else self.nodes[i].radius) * (1 + eps)]
    
    def search_nodes(self, x, eps=0):
        return [self.nodes[i] for i in range(len(self.nodes)) if norm(self.nodes[i].coord - x) < 
                (self.rads[i] if self.rads is not None else self.nodes[i].radius) * (1 + eps)]

class KDTreeSupportNodeSearcher:
    
    def __init__(self, nodes, rads=None, loosen=None):
        self.nodes = nodes
        if rads is None:
            rads = [node.radius for node in nodes]
        self.rads = rads
        
        if loosen is None:
            self.loosen = self.estimate_loosen(rads)
        elif isinstance(loosen, Number):
            self.loosen = loosen
        else:
            self.loosen = loosen(nodes, rads)
        
        self.coords = np.array([node.coord for node in nodes], dtype=float)
        self._build_kd_tree(rads)
    
    def estimate_loosen(self, rads):
        mean = np.mean(rads)
        std = np.std(rads)
        return mean + std
    
    def _build_kd_tree(self, rads):
        keys = []
        medium_indes = []
        loosen = self.loosen
        coords = self.coords
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
            self.medium_indes = np.zeros((len(keys), 2))
            self.medium_indes[:, 0] = medium_indes
            self._current_generation = 1
        self.kd_tree = KDTree(keys)
                
    def search_indes(self, x, eps=0):
        r = self.loosen * (1 + eps)
        indes = self.kd_tree.query_ball_point(x, r, p=float('inf'), eps=eps)
        if self.medium_indes is not None:
            self._shift_current_generation()
            medium_indes = self.medium_indes
            current_generation = self._current_generation
            result = []
            for i in indes:
                index, generation = medium_indes[i]
                if generation == current_generation:
                    continue
                
                medium_indes[i, 1] = current_generation
                rad = self.rads[index]
                coord = self.coords[index]
                if np.linalg.norm(x - coord) >= rad * (1 + eps):
                    continue
                result.append(index)
            return result
        else:
            return [i for i in indes if np.linalg.norm(self.coords[i] - x) < self.rads[i] * (1 + eps)]
        
    def search_nodes(self, x, eps=0):
        return [self.nodes[i] for i in self.search_indes(x, eps)]
    
    def _shift_current_generation(self):
        if self._current_generation == sys.maxsize:
            self._current_generation = 0
            self.medium_indes[:, 1] = -1
        else:
            self._current_generation += 1     
