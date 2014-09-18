'''

@author: "epsilonyuan@gmail.com"
'''

import numpy as np
from pymfr.misc.tools import ensure_sequence
from pymfr.model.raw_model import MFNode
import itertools

def link_segments(pred, succ):
    pred.succ = succ
    succ.pred = pred

def is_clockwise(head):
    negs = 0
    poss = 0
    for seg in head:
        succ = seg.succ
        cross = np.cross(
           seg.end.coord - seg.start.coord,
           succ.end.coord - succ.start.coord)
        if cross > 0:
            poss += 1
        elif cross < 0:
            negs += 1
    if negs > poss:
        return True
    elif negs < poss:
        return False
    else:
        raise ValueError()
    
def is_chain_linking_well(head):
    for seg in head:
        if seg.succ is None or seg.succ.pred != seg:
            return False
    return True

def create_linked_segments_by_coords(coords, closed=True, node_type=None):
    if node_type is None:
        node_type = MFNode
    return create_linked_segments_by_nodes(itertools.groupby(coords, node_type))
    

def create_linked_segments_by_nodes(nodes, closed=True):
    segs = [LinkedSegment(node) for node in nodes]
    segs_size = len(segs)
    for i in range(segs_size if closed else segs_size - 1):
        link_segments(segs[i], segs[(i + 1) % segs_size])
    return segs

class ChainLinkError(Exception):
    def __init__(self, head):
        self.seg = head

class ChainClockwiseError(Exception):
    def __init__(self, polygon, chain):
        self.polygon = polygon
        self.chain = chain

class LinkedSegment:
    
    def __init__(self, start=None, pred=None, succ=None):
        self.start = start
        self.pred = pred
        self.succ = succ
        
    @property
    def end(self):
        return self.succ.start
    
    @end.setter
    def end(self, node):
        self.succ.start = node
        
    def connect(self, seg, as_succ=True):
        if as_succ:
            seg.pred = self
            self.succ = seg
        else:
            seg.succ = self
            self.pred = seg
    
    def __iter__(self):
        return LinkedSegmentIterator(self)
    
class LinkedSegmentIterator:
    
    def __init__(self, start):
        self.start = start
        self.next = start
        
    def __next__(self):
        if self.next == None:
            raise StopIteration
        result = self.next
        self.next = self.next.succ
        if self.next == self.start:
            self.next = None
        return result

class SimpRangle:
    
    def __init__(self, coords=None):
        self.coords = coords
        
class Facet:
    
    @classmethod
    def create_by_coordss(cls, coordss):
        heads = [create_linked_segments_by_coords(coords)[0] for coords in coordss]
        return Facet(heads)
    
    def __init__(self, heads=None):
        self.heads = ensure_sequence(heads)
    
    def check(self):
        self.check_chains_link()
        self.check_chains_clockwise()
    
    def check_chains_link(self):
        for head in self.heads:
            if not is_chain_linking_well(head):
                raise ChainLinkError(head) 

    def check_clockwise(self):
        exp = False
        for head in self.heads:
            if exp != is_clockwise(head):
                raise ChainClockwiseError(self, head)
            exp = True
