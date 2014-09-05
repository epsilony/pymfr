'''
@author: "epsilonyuan@gmail.com"
'''

import numpy as np
from pymfr.misc.math import partial_size
from scipy.linalg import cho_solve, cho_factor

class MLSRK:
    """
    the important fields:
    
    weight_func:
        callable, weight function
        
        to get the weight $$w_i(x)$$ at position `x`
        of node which has an assembly index `index`:
        >>> some_config_some_weight_func(weight_func,all_nodes_information)
        ...
        >>> weight_func.spatial_order=spatial_order 
        >>> weight_func.partial_order=partial_order
        ...
        >>> w = weight_func(x,index)
        
        where w[j]=$$w_{i,j}(x)$$
        
        w.shape: (partial_size,)
        x.shape: (spatial_dim,)
        
        required properties for weight_func:
        weight_func.partial_order
        weight_func.spatial_dim
        
    kernel:
        callable, kernel function
        
        to get the kernel function at y when kernel center is x:
        >>>kernel.partial_order=partial_order
        >>>kernel.center=x
        >>>vals=kernel(y)
        
        where vals.shape: (kernel_size,partial_size)
        
    As:
        moment matrix A and its partial derivatives
        As[j]: $$A_{,j}$$
        len(As): >= current partial_size
        As[j].shape: (kernel_size,kernel_size)
    
    gammas:
        medium vectors for hack of avoiding inversing As
        len(gammas): >= current partial_size
        gammas[j]: $$\gamma_{,j}$$
        gammas[j].shape: (kernel_size,)
    
    Bs:
        column i are weighted kernel vector or node i of support_domain
        (note: here i is not node's assembly index)
        for speed up, let it be a view of _Bs (reuse _Bs):
        Bs.flags.f_continous: True
        Bs[j].shape: (kernel_size,nodes_size)
        len(Bs[j]): >= current partial_size
    _Bs:
        len(_Bs): >= current partial_size
        _Bs[j].shape[0]: kernel_size
        _Bs[j].shape[1]: >= current nodes_size 
    """
    
    
    def __init__(self, weight_func, kernel, spatial_dim=2, partial_order=1):
        self.weight_func = weight_func
        self.kernel = kernel

        if not weight_func or not kernel:
            raise ValueError()
        self.As = None
        self.gammas = None
        self.Bs = None
        self._Bs = None
        self._t_vec=None
        self.spatial_dim = spatial_dim
        self.partial_order = partial_order
        
    def calc(self, x, node_coords, node_indes, out=None):
        partial_size = self.partial_size()
        kernel = self.kernel
        weight_func = self.weight_func
        kernel_size = kernel.size
        
        kernel.partial_order = 0
        weight_func.partial_order = self.partial_order
        
        nodes_size = len(node_indes)
        if nodes_size != len(node_coords):
            raise ValueError("node_coords and node_indes size mismatch")

        self._init_matrice(nodes_size)
        As = self.As
        Bs = self.Bs
        for A in As:
            A.fill(0)
        for B in Bs:
            B.fill(0)
        gammas = self.gammas
        
        m = np.empty((kernel_size,kernel_size))

        for i in range(nodes_size):
            coord = node_coords[i]
            index = node_indes[i]
            
            p_i = kernel(coord)
            np.dot(p_i.reshape((-1, 1)), p_i.reshape((1, -1)), m)
            
            weights = weight_func(x, index)
            for j in range(partial_size):
                As[j] += weights[j] * m
                Bs[j][:, i] = p_i * weights[j]
        
        kernel.partial_order = self.partial_order
        ps = kernel(x)
        
        cl = cho_factor(As[0])
        gamma = gammas[0]
        np.copyto(gamma,ps[0])
        gamma = cho_solve(cl, gamma, overwrite_b=True)
        
        if not out:
            out = np.empty((partial_size,nodes_size))
        gamma.dot(Bs[0], out=out[0])
        
        t=self._t_vec
        
        for j in range(1, partial_size):
            
            gamma_j = gammas[j]
            gamma_j.fill(0)
            
            As[j].dot(gamma, out=t)
            gamma_j += ps[j]
            gamma_j -= t
            
            gamma_j = cho_solve(cl, gamma_j, overwrite_b=True)
            t2=gamma_j.dot(Bs[0])
            
            out_j = out[j]
            gamma.dot(Bs[j], out=out_j)
            
            out_j += t2
        return out

    def _init_matrice(self, nodes_size):
        kernel_size = self.kernel.size
        As = self.As
        gammas = self.gammas
        _Bs = self._Bs
        Bs=self.Bs
        partial_size = self.partial_size()
        if As is None or len(As) < partial_size or As[0].shape[0] != kernel_size:
            As = [np.empty((kernel_size, kernel_size)) for _i in range(partial_size)]
            self.As = As
        
        if _Bs is None or len(_Bs) < partial_size or _Bs[0].shape[0]!=kernel_size or _Bs[0].shape[1]<nodes_size:
            _Bs = [np.empty((kernel_size, nodes_size), order='F') for _i in range(partial_size)]
            self._Bs = _Bs
            Bs=None
        
        if Bs is None or len(Bs)< partial_size:
            Bs=[_Bs[j][:,:nodes_size] for j in range(partial_size)]
            self.Bs=Bs
        elif Bs[0].shape!=(kernel_size,nodes_size):
            for j in range(partial_size):
                Bs[j]=_Bs[j][:,:nodes_size] 
        
        if self._t_vec is None or self._t_vec.shape[0] !=kernel_size:
            self._t_vec=np.empty((kernel_size,)) 
        
        if gammas is None or len(gammas) < partial_size or gammas[0].shape != (kernel_size,):
            gammas = np.zeros((partial_size,kernel_size))
            self.gammas = gammas

    def partial_size(self):
        return partial_size(self.spatial_dim, self.patial_order)

    @property
    def spatial_dim(self):
        return self._spatial_dim
    
    @spatial_dim.setter
    def spatial_dim(self, spatial_dim):
        self._spatial_dim = spatial_dim
        self.kernel.spatial_dim = spatial_dim
        self.weight_func.spatial_dim = spatial_dim
            #self._del_matrice()
    
    @property
    def partial_order(self):
        return self._partial_order
    
    @partial_order.setter
    def partial_order(self, value):
        if value < 0 or value > 1:
            raise ValueError('only supports partial_order 0 or 1')
        self._partial_order = value
        #self._del_matrice()
    

        
        
