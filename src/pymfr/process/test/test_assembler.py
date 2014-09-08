'''

@author: "epsilonyuan@gmail.com"
'''

import numpy as np
from pymfr.process.assembler import VirtualLoadWorkAssembler, LagrangleDirichletLoadAssembler, \
    MechanicalVolumeAssembler2D, PoissonVolumeAssembler
from nose.tools import assert_almost_equal, ok_

def test_virtual_load_assembler():
    for data in [_data_virtual_load_assembler_1d(),
                 _data_virtual_load_assembler_2d()
                 ]:
        assembler = VirtualLoadWorkAssembler(vector=data['vector'], value_dim=data['value_dim'])
        for weight, indes, test_shape_func, load, exp in zip(*[data[name] for name in 'weights indess test_shape_funcs loads exp_vectors'.split()]):
            assembler.assemble(weight, indes, test_shape_func, load)
            assert_almost_equal(0, np.linalg.norm(assembler.vector - exp))

def _data_virtual_load_assembler_1d():
    value_dim = 1
    vector = np.zeros(5)
    
    weights = 0.2, 0.4
    
    test_shape_funcs = (
                      [[1, 2, 3]],
                      [[4, 5]],
                      )
    test_shape_funcs = [np.array(f, dtype=float) for f in test_shape_funcs]
    
    indess = ([4, 0, 1], [3, 1])
    
    loads = ([2], [-3])
    loads = [np.array(l, dtype=float) for l in loads]
    
    exp_vectors = (
        [0.8, 1.2, 0, 0, 0.4],
        [0.8, -4.8, 0, -4.8, 0.4]
        )
    exp_vectors = [np.array(v, dtype=float) for v in exp_vectors]
    
    return {'value_dim':value_dim,
            'vector':vector,
            'weights':weights,
            'test_shape_funcs':test_shape_funcs,
            'indess':indess,
            'loads':loads,
            'exp_vectors':exp_vectors
            }

def _data_virtual_load_assembler_2d():
    value_dim = 2
    vector = np.zeros((2 * 3,))
    
    weights = [0.2, 0.5, 0.1]
    test_shape_funcs = (
                      [[-1, -2]],
                      [[3]],
                      [[2]],
                      )
    indess = (
            [2, 0],
            [2, ],
            [1, ],
            )
    loads = ([-1, 2], [3, -1], [2, -3])
    exp_vectors = (
         [0.4, -0.8, 0, 0, 0.2, -0.4],
         [0.4, -0.8, 0, 0, 4.7, -1.9],
         [0.4, -0.8, 0.4, -0.6, 4.7, -1.9],
                 )
    
    result = {'value_dim':value_dim,
            'vector':vector,
            'weights':weights,
            'test_shape_funcs':test_shape_funcs,
            'indess':indess,
            'loads':loads,
            'exp_vectors':exp_vectors
            }
    
    for name in ['test_shape_funcs', 'loads', 'exp_vectors']:
        ds = result[name]
        result[name] = [np.array(d, dtype=float) for d in ds]
    
    return result

def test_lagrangle_dirichlet_assembler():
    for data in [_lagrangle_dirichlet_data_1d(), _lagrangle_dirichlet_data_2d()]:
        _test_lagrangle_dirichlet_assembler(data)

def _test_lagrangle_dirichlet_assembler(data):
    assembler = LagrangleDirichletLoadAssembler(
                   matrix=data['matrix'],
                   vector=data['vector'],
                   value_dim=data['value_dim'],
                   lagrangle_nodes_size=data['lagrangle_nodes_size']
                   )
    
    for (weight,
         node_indes,
         test_shape_func,
         trial_shape_func,
         lagrangle_test_shape_func,
         lagrangle_trial_shape_func,
         lagrangle_node_indes,
         load,
         load_validity,
         exp_mat,
         exp_vec
         ) in zip(*[
                        data[name] for name in [
                                                'weights',
                                                'node_indess',
                                                'test_shape_funcs',
                                                'trial_shape_funcs',
                                                'lagrangle_test_shape_funcs',
                                                'lagrangle_trial_shape_funcs',
                                                'lagrangle_node_indess',
                                                'loads',
                                                'load_validities',
                                                'exp_mats',
                                                'exp_vecs',
                                                ]
                    ]):
        assembler.assemble(weight, node_indes, test_shape_func, load, load_validity, lagrangle_node_indes, lagrangle_test_shape_func, trial_shape_func, lagrangle_trial_shape_func)
        ok_(np.max(np.abs(assembler.vector - exp_vec)) < 1e-6)
        ok_(np.max(np.abs(assembler.matrix - exp_mat)) < 1e-6)

def _lagrangle_dirichlet_data_1d():
    result = {'value_dim':1,
            'lagrangle_nodes_size':3,
            'matrix':np.zeros((6, 6)),
            'vector':np.zeros((6,)),
            'weights':[1000, 0.1, 2, ],
            'test_shape_funcs':(
               [[1000]],
               [[0.2, 0.3]],
               [[0.4]],

               ),
            'trial_shape_funcs':(
                None,
                None,
                [[0.5]],
            ),
            'node_indess':(
                [0],
                [1, 2],
                [1],
                            ),
            'lagrangle_node_indess':(
                [3],
                [5, 3],
                [3]
                                    ),
            'lagrangle_test_shape_funcs':(
                [[1000]],
                [[-0.1, 0.2]],
                [[-0.3]],
                                          ),
            'lagrangle_trial_shape_funcs':(
                [[1000]],
                [[0.2, 0.4]],
                None,
                                           ),
            'loads':(
                [1000],
                [0.5],
                [-0.5],
                     ),
            'load_validities':(
                [False],
                [True],
                [True],
                               ),
            'exp_mats':([[0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1]
                        ],
                        [[0, 0, 0, 0, 0, 0, ],
                        [0, 0, 0, 0.008, 0, 0.004],
                        [0, 0, 0, 0.012, 0, 0.006],
                        [0, 0.004, 0.006, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, -0.002, -0.003, 0, 0, 0]
                        ],
                        [[0, 0, 0, 0, 0, 0],
                        [0, 0, 0, -0.232, 0, 0.004],
                        [0, 0, 0, 0.012, 0, 0.006],
                        [0, -0.296, 0.006, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, -0.002, -0.003, 0, 0, 0]
                        ],
                        ),
            'exp_vecs':([0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0.01, 0, -0.005],
                        [0, 0, 0, 0.31, 0, -0.005],)
            }
    for name in ['loads',
                 'lagrangle_trial_shape_funcs',
                 'lagrangle_test_shape_funcs',
                 'test_shape_funcs',
                 'trial_shape_funcs',
                 'exp_mats',
                 'exp_vecs'
                 ]:
        result[name] = [np.array(d, dtype=float) if d is not None else None for d in result[name]]
    return result

def _lagrangle_dirichlet_data_2d():
    result = {'value_dim':2,
            'lagrangle_nodes_size':3,
            'matrix':np.zeros((12, 12)),
            'vector':np.zeros((12,)),
            'weights':[1000, 0.1, 2, ],
            'test_shape_funcs':(
               [[1000]],
               [[0.2, 0.3]],
               [[0.4]],

               ),
            'trial_shape_funcs':(
                None,
                None,
                [[0.5]],
            ),
            'node_indess':(
                [0],
                [1, 2],
                [1],
                            ),
            'lagrangle_node_indess':(
                [3],
                [5, 3],
                [3]
                                    ),
            'lagrangle_test_shape_funcs':(
                [[1000]],
                [[-0.1, 0.2]],
                [[-0.3]],
                                          ),
            'lagrangle_trial_shape_funcs':(
                [[1000]],
                [[0.2, 0.4]],
                None,
                                           ),
            'loads':(
                [1000, 1000],
                [0.5, 1],
                [-0.5, 2],
                     ),
            'load_validities':(
                [False, False],
                [True, False],
                [True, True],
                               ),
            'exp_mats':([
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                         ],
                        [
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0.008, 0, 0, 0, 0.004, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0.012, 0, 0, 0, 0.006, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0.004, 0, 0.006, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, -0.002, 0, -0.003, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                         ],
                        [
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, -0.232, 0, 0, 0, 0.004, 0],
                         [0, 0, 0, 0, 0, 0, 0, -0.24, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0.012, 0, 0, 0, 0.006, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, -0.296, 0, 0.006, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, -0.3, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, -0.002, 0, -0.003, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                         ],
                        ),
            'exp_vecs':([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0.01, 0, 0, 0, -0.005, 0],
                        [0, 0, 0, 0, 0, 0, 0.31, -1.2, 0, 0, -0.005, 0],)
            }
    for name in ['loads',
                 'lagrangle_trial_shape_funcs',
                 'lagrangle_test_shape_funcs',
                 'test_shape_funcs',
                 'trial_shape_funcs',
                 'exp_mats',
                 'exp_vecs'
                 ]:
        result[name] = [np.array(d, dtype=float) if d is not None else None for d in result[name]]
    return result

def test_mechanical_2d_assembler():
    
    data = {
          'matrix':np.ones((6, 6)),
          'test_shape_funcs':(
                             [[1000, 1000],
                              [0.1, 0.2],
                              [0.3, 0.4]],
                             ),
          'trial_shape_funcs':(
                              [[1000, 1000],
                               [0.5, 0.6],
                               [0.7, 0.8],
                                ],
                              ),
          'constitutive_law':[[0.2, 0.3, 0.4],
                              [0.5, 0.6, 0.7],
                              [0.8, 0.9, 1.0],
                              ],
          'weights':[0.5],
          'indess':([2, 0],),
          'exp_mats':(np.array([
                      [0.3, 0.312, 0, 0, 0.258, 0.267],
                      [0.3, 0.312, 0, 0, 0.258, 0.267],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0.214, 0.222, 0, 0, 0.184, 0.19],
                      [0.193, 0.201, 0, 0, 0.166, 0.172]
                       ], dtype=float) + 1,
                      )
          }
    
    for name in ['test_shape_funcs', 'trial_shape_funcs']:
        data[name] = [np.array(d, dtype=float) if d is not None else None for d in data[name]]
    
    assembler = MechanicalVolumeAssembler2D(data['matrix'])
    constitutive_law = np.array(data['constitutive_law'], dtype=float)
    for weight, indes, test_shape_func, trial_shape_func, exp_mat in zip(*[data[name] for name in [
                       'weights',
                       'indess',
                       'test_shape_funcs',
                       'trial_shape_funcs',
                       'exp_mats'
                       ]]):
        
        assembler.assemble(weight, indes, test_shape_func, constitutive_law, trial_shape_func)
        delta = np.abs(assembler.matrix - exp_mat)
        ok_(np.max(delta) < 1e-6)

def test_poission_2d():
    data = {
          'value_dim':2,
          'matrix':np.ones((4, 4)),
          'test_shape_funcs':(
                             [[1000, 1000],
                              [0.1, 0.2],
                              [0.3, 0.4]],
                             ),
          'trial_shape_funcs':(
                              [[1000, 1000],
                               [0.5, 0.6],
                               [0.7, 0.8],
                                ],
                              ),
          'weights':[0.5],
          'indess':([2, 0],),
          'exp_mats':(np.array([[0.22, 0, 0.19, 0],
                                [0, 0, 0, 0],
                                [0.15, 0, 0.13, 0],
                                [0, 0, 0, 0]
                       ], dtype=float) + 1,
                      )
          }
    
    for name in ['test_shape_funcs', 'trial_shape_funcs']:
        data[name] = [np.array(d, dtype=float) if d is not None else None for d in data[name]]
    
    assembler = PoissonVolumeAssembler(data['matrix'])
    for weight, indes, test_shape_func, trial_shape_func, exp_mat in zip(*[data[name] for name in [
                       'weights',
                       'indess',
                       'test_shape_funcs',
                       'trial_shape_funcs',
                       'exp_mats'
                       ]]):
        
        assembler.assemble(weight, indes, test_shape_func, trial_shape_func)
        delta = np.abs(assembler.matrix - exp_mat)
        ok_(np.max(delta) < 1e-6)
