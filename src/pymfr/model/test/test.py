'''

@author: "epsilonyuan@gmail.com"
'''

import numpy as np
from pymfr.model.searcher import KDTreeNodeSearcher, RawNodeSearcher, RawSegmentSearcher, KDTreeSegmentSearcher
from pymfr.misc.tools import twod_uniform_coords, rand_coords
from nose.tools import assert_set_equal, eq_
from math import pi

class MockNode:
    def __init__(self, coord):
        self.coord = coord

def test_kdtree_nodes_searcher():
    xs = np.linspace(-1, 8, 10)
    ys = np.linspace(-5, 4, 10)
    
    coords = twod_uniform_coords(xs, ys)
    nodes = [MockNode(coord) for coord in coords]
    searcher = KDTreeNodeSearcher(node for node in nodes)
    
    for x, rad, eps, exp_indes in ([(-0.5, -4.5), 1, 0, (0, 1, 10, 11)],
                                [(-0.5, -4.5), 2 ** 0.5 / 2, 0.001, (0, 1, 10, 11)],
                                [(-0.5, -4.5), 1.5 * 2 ** 0.5, 0.001, (0, 1, 2, 10, 11, 12, 20, 21, 22)]):
        act_indes = searcher.search_indes(x, rad, eps)
        act_nodes = searcher.search(x, rad, eps)
        exp_nodes = [nodes[i] for i in exp_indes]
        eq_(len(act_nodes), len(exp_nodes))
        assert_set_equal(set(exp_indes), set(act_indes))
        
def test_kdtree_raw_comp_nodes_searcher():
    nodes_size = 100
    coords = np.random.rand(nodes_size, 2)
    coords[:, 0] = -2 + coords[:, 0] * 9
    coords[:, 1] = -4 + coords[:, 0] * 9
    
    nodes = [MockNode(coord) for coord in coords]
    kd_searcher = KDTreeNodeSearcher([node for node in nodes])
    raw_searcher = RawNodeSearcher(node for node in nodes)
    
    
    xs_size = 20
    xs = np.random.rand(xs_size, 2)
    xs[:, 0] = -2 + xs[:, 0] * 9
    xs[:, 1] = -4 + xs[:, 1] * 9
    
    rads = np.random.rand(xs_size) * 5
    
    for x, rad in zip(xs, rads):
        kd_indes = kd_searcher.search_indes(x, rad)
        raw_indes = raw_searcher.search_indes(x, rad)
        assert_set_equal(set(kd_indes), set(raw_indes))
        
        kd_nodes = kd_searcher.search(x, rad)
        raw_nodes = raw_searcher.search(x, rad)
        assert_set_equal(set(kd_nodes), set(raw_nodes))

class MockSegment:
    def __init__(self, startcoord, endcoord):
        self.start = MockNode(startcoord)
        self.end = MockNode(endcoord)
        
def test_raw_segment_searcher():
    xs = np.linspace(1, 10, 10)
    ys = np.linspace(1, 10, 10)
    coords = twod_uniform_coords(xs, ys)
    
    segment_start_end_indes = [
         (0, 1),
         (10, 11),
         (20, 21),
         (30, 31),
         (40, 41),
         (2, 12),
         (32, 23),
         (42, 33),
         (51, 42)
         ]
    
    segments = [MockSegment(coords[start], coords[end]) for start, end in segment_start_end_indes]
    searcher = RawSegmentSearcher(segments)
    
    for x, rad, eps, exp_indes in [
        ([1.5, 1.5], 0.5, 1e-6, [0, 1]),
        ([1.5, 1.5], 0.49, 0, []),
        ([1.5, 1.5], 1.5, 1e-6, [0, 1, 2, 5]),
        ([1.5, 1.5], 1.49, 0, [0, 1]),
        ([1.5, 1.5], 2 * 2 ** 0.5, 1e-6, [0, 1, 2, 3, 5, 6]),
        ([1.5, 1.5], 1.99 * 2 ** 0.5, 0, [0, 1, 2, 3, 5]),
        ([1.5, 1.5], 2.5 * 2 ** 0.5, 1e-6, [0, 1, 2, 3, 4, 5, 6, 7]),
        ([1.5, 1.5], 2.49 * 2 ** 0.5, 0, [0, 1, 2, 3, 4, 5, 6]),
        ]:
        act_indes = searcher.search_indes(x, rad, eps)
        assert_set_equal(set(exp_indes), set(act_indes))
        
        act_segs = searcher.search(x, rad, eps)
        exp_segs = [segments[i] for i in exp_indes]
        assert_set_equal(set(exp_segs), set(act_segs))
        
def test_raw_kdtree_segment_searcher_cmp():
    start_range = np.array([[-10, -10], [90, 90]], dtype=float)
    length_mean = 20
    length_std = 5
    num_segs = 100
    
    segments = gen_rand_segments(start_range, length_mean, length_std, num_segs)
    raw_searcher = RawSegmentSearcher(segments)
    kd_searcher = KDTreeSegmentSearcher(segments)
    
    pts_num = 50
    pts = rand_coords(start_range, pts_num)
    rads = rand_coords(np.array([[5], [60]], dtype=float),pts_num)
    
    eps = 1e-6
    for pt, rad in zip(pts,rads):
        raw_indes = raw_searcher.search_indes(pt, rad, eps)
        kd_indes = kd_searcher.search_indes(pt, rad, eps)
        assert_set_equal(set(raw_indes), set(kd_indes))
        
        raw_segs = raw_searcher.search(pt, rad, eps)
        kd_segs = kd_searcher.search(pt, rad, eps)
        assert_set_equal(set(raw_segs),set(kd_segs))

def gen_rand_segments(start_range, length_mean, length_std, num):
    start_coords = rand_coords(start_range, num)
    lengths = np.random.randn(num) * length_std + length_mean
    
    vec_arz = np.random.rand(num) * (2 * pi)
    vec = np.empty((num, 2))
    vec[:, 0] = np.cos(vec_arz) * lengths
    vec[:, 1] = np.sin(vec_arz) * lengths
    
    end_coords = start_coords + vec
    
    return [MockSegment(start_coord, end_coord) for start_coord, end_coord in zip(start_coords, end_coords)]
    
    
