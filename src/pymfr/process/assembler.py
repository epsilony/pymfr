'''
@author: "epsilonyuan@gmail.com"
'''
import numpy as np
from pymfr.misc.tools import add_to_2d
from injector import Key
from pymfr.misc.mixin import SetupMixin

class VirtualLoadWorkAssembler(SetupMixin):
    __prerequisites__ = ['vector', 'value_dim']
    
    def assemble(self, weight, node_indes, test_shape_func, load):
        phi = test_shape_func[0]
        if self.value_dim > 1:
            matrix_indes = np.array(node_indes, dtype=int)
            matrix_indes *= self.value_dim
        else:
            matrix_indes = node_indes
        for i in range(self.value_dim):
            if i != 0:
                matrix_indes += 1
            self.vector[matrix_indes] += phi * (load[i] * weight)

class LagrangleDirichletLoadAssembler(SetupMixin):
    
    __prerequisites__ = "matrix, vector, value_dim, lagrangle_nodes_size".split(", ")
    __after_setup__ = ['set_matrix_lagrangle_diags']   
     
    def set_matrix_lagrangle_diags(self, **kwargs):
        for i in range(-self.lagrangle_nodes_size * self.value_dim, 0):
            self.matrix[i, i] = 1
        
        return self
 
    def assemble(self, weight, node_indes, test_shape_func, load, load_validity, lagrangle_node_indes, lagrangle_test_shape_func, trial_shape_func=None, lagrangle_trial_shape_func=None):
        phi_left = test_shape_func[0]
        phi_right = trial_shape_func[0] if trial_shape_func is not None else phi_left
        lambda_left = lagrangle_test_shape_func[0]     
        
        g_left = weight * np.dot(lambda_left.reshape((-1, 1)), phi_right.reshape((1, -1)))
        
        if lagrangle_trial_shape_func is None and trial_shape_func is None:
            g_right = g_left.T
        else:
            lambda_right = lambda_left if lagrangle_trial_shape_func is None else lagrangle_trial_shape_func[0]
            g_right = weight * np.dot(phi_left.reshape((-1, 1)), lambda_right.reshape((1, -1)))
        
        if self.value_dim > 1:
            matrix_indes = np.array(node_indes, dtype=int)
            matrix_indes *= self.value_dim
            lagrangle_matrix_indes = np.array(lagrangle_node_indes, dtype=int)
            lagrangle_matrix_indes *= self.value_dim    
        else:
            matrix_indes = node_indes
            lagrangle_matrix_indes = lagrangle_node_indes
        
        for i in range(self.value_dim):
            if i != 0:
                matrix_indes += 1
                lagrangle_matrix_indes += 1
            if not load_validity[i]:
                continue
            add_to_2d(self.matrix, lagrangle_matrix_indes, matrix_indes, g_left)
            add_to_2d(self.matrix, matrix_indes, lagrangle_matrix_indes, g_right)
            self.vector[lagrangle_matrix_indes] += lambda_left * (weight * load[i])
            self.matrix[lagrangle_matrix_indes, lagrangle_matrix_indes] = 0
    
class PenaltyDirichletLoadAssembler(SetupMixin):
    __prerequisites__ = "matrix, vector, value_dim".split(", ")
    __optionals__ = [('penalty', 1e6)]
        
    def assemble(self, weight, nodes_indes, test_shape_func, load, load_validity, trial_shape_func=None):
        phi_left = test_shape_func[0]
        phi_right = phi_left if trial_shape_func is None else trial_shape_func[0]
        
        if self.value_dim > 1:
            matrix_indes = np.array(nodes_indes, dtype=int)
            matrix_indes *= self.value_dim
        
        for i in range(self.value_dim):
            if i != 0:
                matrix_indes += 1
            if not load_validity[i]:
                continue
            factor = weight * self.penalty
            matrix_value = factor * np.dot(phi_left.reshape((-1, 1)), phi_right.reshape((1, -1)))
            add_to_2d(self.matrix, matrix_indes, matrix_indes, matrix_value)
            self.vector[matrix_indes] += phi_left * (factor * load[i])

class MechanicalVolumeAssembler2D(SetupMixin):
    __prerequisites__ = ['matrix']
        
    def assemble(self, weight, nodes_indes, test_shape_func, constitutive_law, trial_shape_func=None):
        
        b_left = np.zeros((len(nodes_indes) * 2, 3))
        b_left[::2, 0] = test_shape_func[1]
        b_left[1::2, 1] = test_shape_func[2]
        b_left[::2, 2] = test_shape_func[2]
        b_left[1::2, 2] = test_shape_func[1]
        
        if trial_shape_func is None:
            b_right = b_left.T
        else:
            b_right = np.zeros((3, len(nodes_indes) * 2))
            b_right[0, ::2] = trial_shape_func[1]
            b_right[1, 1::2] = trial_shape_func[2]
            b_right[2, ::2] = trial_shape_func[2]
            b_right[2, 1::2] = trial_shape_func[1]
        
        matrix_indes = np.empty(2 * len(nodes_indes), dtype=int)
        matrix_indes[::2] = nodes_indes
        matrix_indes[::2] *= 2
        matrix_indes[1::2] = matrix_indes[::2]
        matrix_indes[1::2] += 1
        
        matrix_value = b_left.dot(constitutive_law).dot(b_right) * weight
        add_to_2d(self.matrix, matrix_indes, matrix_indes, matrix_value)
        
class PoissonVolumeAssembler(SetupMixin):
    __prerequisites__ = ['matrix']
    
    def assemble(self, weight, indes, test_shape_func, trial_shape_func=None):
        if trial_shape_func is None:
            trial_shape_func = test_shape_func
        
        mm1, mm2 = self._get_medium_matrice(len(indes))
        mm1.fill(0)
        for i in range(1, len(test_shape_func)):
            phi_left = test_shape_func[i]
            phi_right = trial_shape_func[i]
            np.dot(phi_left.reshape((-1, 1)), phi_right.reshape((1, -1)), mm2)
            np.add(mm1, mm2, mm1)
        mm1 *= weight
        add_to_2d(self.matrix, indes, indes, mm1)
        
    def _get_medium_matrice(self, size):
        return np.empty((size, size)), np.empty((size, size))
