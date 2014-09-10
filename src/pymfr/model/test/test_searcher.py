'''

@author: "epsilonyuan@gmail.com"
'''

import numpy as np
from pymfr.model.searcher import KDTreeNodeSearcher, RawNodeSearcher, RawSegmentSearcher, KDTreeSegmentSearcher, \
    RawSupportNodeSearcher, KDTreeSupportNodeSearcher, VisibleSupportNodesSearcher2D
from pymfr.misc.tools import twod_uniform_coords, rand_coords
from nose.tools import assert_set_equal, eq_, assert_almost_equal, ok_
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
        eq_(len(kd_nodes), len(raw_nodes))
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
        eq_(len(exp_segs), len(act_segs))
        assert_set_equal(set(exp_segs), set(act_segs))
        
def test_raw_kdtree_segment_searcher_cmp():
    start_range = np.array([[-10, -10], [90, 90]], dtype=float)
    length_mean = 10
    length_std = 20
    num_segs = 100
    
    segments = gen_rand_segments(start_range, length_mean, length_std, num_segs)
    raw_searcher = RawSegmentSearcher(segments)
    kd_searcher = KDTreeSegmentSearcher(segments)
    
    pts_num = 200
    pts = rand_coords(start_range, pts_num)
    rads = rand_coords(np.array([[5], [60]], dtype=float), pts_num)
    
    eps = 1e-6
    for pt, rad in zip(pts, rads):
        raw_indes = raw_searcher.search_indes(pt, rad, eps)
        kd_indes = kd_searcher.search_indes(pt, rad, eps)
        eq_(len(raw_indes), len(kd_indes))
        assert_set_equal(set(raw_indes), set(kd_indes))
        
        raw_segs = raw_searcher.search(pt, rad, eps)
        kd_segs = kd_searcher.search(pt, rad, eps)
        eq_(len(raw_segs), len(kd_segs))
        assert_set_equal(set(raw_segs), set(kd_segs))

def gen_rand_segments(start_range, length_mean, length_std, num):
    start_coords = rand_coords(start_range, num)
    lengths = np.random.randn(num) * length_std + length_mean
    
    vec_arz = np.random.rand(num) * (2 * pi)
    vec = np.empty((num, 2))
    vec[:, 0] = np.cos(vec_arz) * lengths
    vec[:, 1] = np.sin(vec_arz) * lengths
    
    end_coords = start_coords + vec
    
    return [MockSegment(start_coord, end_coord) for start_coord, end_coord in zip(start_coords, end_coords)]
    
def test_raw_kdtree_cmp_support_node_searcher():
    nodes_size = 100
    coords = np.random.rand(nodes_size, 2)
    coords[:, 0] = -2 + coords[:, 0] * 9
    coords[:, 1] = -4 + coords[:, 0] * 9
    
    nodes = [MockNode(coord) for coord in coords]
    rads = np.random.randn(nodes_size)
    rads = rads + 1
    rads = np.abs(rads)
    
    raw_searcher = RawSupportNodeSearcher(nodes, rads)
    kd_searcher = KDTreeSupportNodeSearcher(nodes, rads)
    
    xs_size = 20
    xs = np.random.rand(xs_size, 2)
    xs[:, 0] = -2 + xs[:, 0] * 9
    xs[:, 1] = -4 + xs[:, 1] * 9
    for x in xs:
        raw_indes = raw_searcher.search_indes(x)
        kd_indes = kd_searcher.search_indes(x)
        eq_(len(raw_indes), len(kd_indes))
        assert_set_equal(set(raw_indes), set(kd_indes))

def test_perturb_center():
    perturb_distance = 0.1
    epsilon = 0.000001
    vss = VisibleSupportNodesSearcher2D(None, None, perturb_distance)
    
    x_bnd_exp_s = (
                 [(0.5, 0.5), [(-0.9 - epsilon, 2.5), (0.5, 0.5), (1.5, 1.2), (0.8 + epsilon, 2.2)], (0.4426537655636671, 0.581923192051904)],  # left end 90deg+
                 [(0.5, 0.5), [(-0.9 + epsilon, 2.5), (0.5, 0.5), (1.5, 1.2), (0.8 + epsilon, 2.2)], (0.5245769576155712, 0.6392694264882368)],  # left end 90deg-
                 [(1.5, 1.2), [(-0.9 + epsilon, 2.5), (0.5, 0.5), (1.5, 1.2), (0.8 + epsilon, 2.2)], (1.4426537655636671, 1.2819231920519041)],  # right end 90deg+
                 [(1.5, 1.2), [(-0.9 + epsilon, 2.5), (0.5, 0.5), (1.5, 1.2), (0.8 - epsilon, 2.2)], (1.3607305735117632, 1.2245769576155712)],  # right end 90deg+
                 [(0.8276927682076163, 0.7293849377453313), [(-0.7 + epsilon, 2.5), (0.5, 0.5), (1.5, 1.2), (0.8 - epsilon, 2.2)], (0.7703465337712835, 0.8113081297972353)],  # at mid of seg
                 [(0.8276927682076163, 0.7293849377453313), [(-0.4571869788019205, 11.458275096364941), (0.5, 0.5), (1.5, 1.2), (0.8 - epsilon, 2.2)], (0.7703465337712835, 0.8113081297972353)],  # near left but not left, deg=60
                 [(0.58192319, 0.55734623), [(-0.4571869788019205, 11.458275096364941), (0.5, 0.5), (1.5, 1.2), (0.8 - epsilon, 2.2)], (0.5845488965157878, 0.6812497837183885)],  # in left angle zone,deg=60
                 [(1.4180768079480959, 1.142653765563667), [(-0.9, 2.5), (0.5, 0.5), (1.5, 1.2), (-0.8956550556943457, 3.0057787389727801)], (1.3298961651561385, 1.2029928717666341)],  # in right angle zone
                 )
    
    for x, bnd_coords, exp in x_bnd_exp_s:
        bnd_coords = np.array(bnd_coords, dtype=float)
        bnd = MockSegment(*bnd_coords[1:3])
        bnd.pred = MockSegment(*bnd_coords[0:2])
        bnd.succ = MockSegment(*bnd_coords[2:4])
        
        act = vss._perturb_x(x, bnd)
        ok_(np.linalg.norm(exp - act) < 1e-6)

def test_visible_support_domain_searcher():
    data = _visible_support_domain_data()
    x_bnd_exp_s = data['x_bnd_exp_s']
    searcher = data['searcher']
    for x, bnd, exp in x_bnd_exp_s:
        act = searcher.search(x, bnd)
        eq_(len(exp), len(act))
        assert_set_equal(set(exp), set(act))

class AnyMock():
    pass

class MockSupportNodesSearcher:
    def search_indes(self, x, eps=0):
        return self.mock_indes

class MockSegmentsSearcher:
    
    def search_indes(self, x, rad, eps=0):
        self.last_rad = rad
        self.last_x = x
        return self.mock_indes

def _visible_support_domain_data():
    
    xs = np.linspace(0, 4, 5)
    ys = np.linspace(0, 4, 5)
    
    coords = twod_uniform_coords(xs, ys)
    
    nodes = [MockNode(coord) for coord in coords]
    
    for i in range(len(nodes)):
        nodes[i].index = i * 2 + 1
    
    segment_indess = [(7, 11, 12, 13, 7), (14, 18, 22), (2, 8)]
    segments = []
    for seg_indes in segment_indess:
        open_chain = []
        for i in range(len(seg_indes) - 1):
            start = nodes[seg_indes[i]]
            end_index = seg_indes[i + 1]
            end = nodes[end_index]
            seg = AnyMock()
            seg.start = start
            seg.end = end
            open_chain.append(seg)
        for i in range(len(open_chain) - 1):
            open_chain[i].succ = open_chain[i + 1]
            open_chain[i + 1].pred = open_chain[i]
        segments.extend(open_chain)
    
    for i in range(len(segments)):
        segments[i].index = i * 2 + 1
        
    x_bnd_exp_s = [([(11, 0.5), (12, 0.5)], 1, [21, 23, 25, 27, 29, 31, 33, 35, 37, 41, 43, 45]),
                 ([(11, 1), (12, 0)], 1, [21, 23, 25, 27, 29, 31, 33, 35, 37, 41, 43, 45]),
                 ([(11, 1), (12, 0)], 2, [21, 23, 25, 27, 29, 31, 33, 35, 37, 41, 43, 45]),
                 ([(17, 0.75), (22, 0.25)], None, [41, 43, 45, 31, 33, 35, 37, 21, 23, 25, 27, 29, 11, 19])]
    t = []
    for x, bnd, exp in x_bnd_exp_s:
        (node_i1, weight1), (node_i2, weight2) = x
        x = coords[node_i1] * weight1 + coords[node_i2] * weight2
        bnd = segments[bnd] if bnd is not None else None
        t.append((x, bnd, exp))
    x_bnd_exp_s = t
    
    support_node_searcher = MockSupportNodesSearcher()
    support_node_searcher.mock_indes = [node.index for node in nodes]
    support_node_searcher.nodes = [None if i % 2 == 0 else nodes[i // 2] for i in range(2 * len(nodes))]
    support_node_searcher.coords = [None if node is None else node.coord for node in support_node_searcher.nodes]
    
    segment_searcher = MockSegmentsSearcher()
    segment_searcher.mock_indes = [segment.index for segment in segments]
    segment_searcher.segments = [None if i % 2 == 0 else segments[i // 2] for i in range(2 * len(segments))]
    
    searcher = VisibleSupportNodesSearcher2D(support_node_searcher, segment_searcher)
    
    t = locals()
    result = {name:t[name] for name in [
                 'x_bnd_exp_s',
                 'searcher',
                 'support_node_searcher',
                 'segment_searcher',
                 'nodes',
                 'coords',
                 'segments'
                 ]}
    return result

    
