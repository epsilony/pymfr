'''

@author: "epsilonyuan@gmail.com"
'''
from collections import Sequence
import numpy as np
from pymfr.misc.mixin import SetupMixin
import collections
from numbers import Number

class FieldProxy:
    def __init__(self, field_name, proxy_name):
        self.field_name = field_name
        self.proxy_name = proxy_name
        
    def __get__(self, obj, owner=None):
        field = getattr(obj, self.field_name)
        return getattr(field, self.proxy_name)
    
    def __set__(self, obj, value):
        field = getattr(obj, self.field_name)
        setattr(field, self.proxy_name, value)
        
def ensure_sequence(obj):
    return obj if isinstance(obj, Sequence) else list(obj)

def twod_uniform_coords(xs, ys):
    xvs, yvs = np.meshgrid(xs, ys)
    
    coords = np.empty((xvs.size, 2))
    coords[:, 0] = xvs.flat
    coords[:, 1] = yvs.flat
    
    return coords

def rand_coords(dst_range, num, rand=None, rand_range=None):
    
    if rand is None:
        rand = np.random.rand
        
    coords = rand(num, dst_range.shape[1])
    return trans_coords(coords, dst_range, rand_range)
    
def trans_coords(coords, dst_range, src_range=None):
    if dst_range.ndim != 2 or dst_range.shape[0] != 2:
        raise ValueError()
    if src_range is None:
        src_range = np.zeros_like(dst_range)
        src_range[1].fill(1)
    elif src_range.shape != dst_range.shape:
        raise ValueError()
    
    r0 = src_range[0].reshape((1, -1))
    r1 = src_range[1].reshape((1, -1))
    
    d0 = dst_range[0].reshape((1, -1))
    d1 = dst_range[1].reshape((1, -1))
    
    coords = (coords - r0) / (r1 - r0) * (d1 - d0) + d0
    return coords

def copy_to_2d(dst, i_indes, j_indes, src):
    src_ndim = np.ndim(src)
    if src_ndim == 0 or (src_ndim == 1 and len(src) == len(j_indes)):
        for i in range(len(i_indes)):
            dst[i_indes[i]][j_indes] = src
    elif src_ndim == 2:
        for i in range(len(i_indes)):
            dst[i_indes[i]][j_indes] = src[i]
    else:
        raise ValueError()

def add_to_2d(dst, i_indes, j_indes, src):
    src_ndim = np.ndim(src)
    if src_ndim == 0 or (src_ndim == 1 and len(src) == len(j_indes)):
        for i in range(len(i_indes)):
            dst[i_indes[i]][j_indes] += src
    elif src_ndim == 2:
        for i in range(len(i_indes)):
            dst[i_indes[i]][j_indes] += src[i]
    else:
        raise ValueError()

def _raw_assign_2d(dst, i_indes, j_indes, src):
    for i in range(len(i_indes)):
        for j in range(len(j_indes)):
            dst[i_indes[i], j_indes[j]] = src[i][j]

def _is_default_filtered(obj):
    if obj is None:
        return True
    return isinstance(obj, (str, Number, bool, type, np.ndarray))

def search_setup_mixins(_root_obj):
    visited = set()
    if _is_default_filtered(_root_obj):
        return
    stack = [_root_obj]
    while stack:
        o = stack.pop()
        if isinstance(o, (collections.Mapping)):
            stack.extend(t for t in o.values() if t is not None and not _is_default_filtered(t))
            continue
        if isinstance(o, (collections.Set, collections.Sequence, collections.MutableSequence)):
            stack.extend(t for t in o if t is not None and not _is_default_filtered(t))
        try:
            o.__hash__()
        except TypeError:
            continue
        if o in visited:
            continue
        if isinstance(o, SetupMixin):
            yield o
        visited.add(o)
        o_dict = getattr(o, '__dict__', None)
        if o_dict is not None:
            stack.extend(t for t in o_dict.values() if t is not None and not _is_default_filtered(t))

def recursively_setup(_root_obj, **kwargs):
    for obj in search_setup_mixins(_root_obj):
        obj.setup(**kwargs)

class SetupStatus:
    def __init__(self):
        self.setables = set()
        self.pres = {}
        self.unset_pres = {}
        self.opts = {}
        self.undefault_opts = {}
        self.unset_opts = {}
    

    
def gen_setup_status(root_obj):
    ss = search_setup_mixins(root_obj)
    ret = SetupStatus()
    for s in ss:
        ret.setables.add(s)
        for name in getattr(s, '__prerequisites__', ()):
            if not name in ret.pres:
                ret.pres[name] = set()
            ret.pres[name].add(s)
            if getattr(s, name, None) is None:
                if not name in ret.unset_pres:
                    ret.unset_pres[name] = set()
                ret.unset_pres[name].add(s)
                
        for name, df in getattr(s, '__optionals__', ()):
            if not name in ret.opts:
                ret.opts[name] = set()
            ret.opts[name].add(s)
            v = getattr(s, name, None)
            if v is None:
                if not name in ret.unset_opts:
                    ret.unset_opts[name] = set()
                ret.unset_opts[name].add(s)
            elif v != df:
                if not name in ret.undefault_opts:
                    ret.undefault_opts[name] = set()
                ret.undefault_opts[name].add(s)
    return ret
