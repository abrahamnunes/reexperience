from patterns import generate_nested_patterns, split_training_test
from rbm import rbm
from viz import viz
import numpy as np
import matplotlib.pyplot as plt

'''
    EXPERIMENTAL PARAMETERS
'''

nvisible  = 200     # number of visible units in networks
nhidden   = 1000    # number of hidden units in networks
nblocks   = 10      # number of lifespan blocks

'''
    GENERATE PATTERNS
'''

pat = generate_nested_patterns(nvisible=nvisible,
                               nclasses=5,
                               nsubclasses=10,
                               npatterns=24,
                               p=0.1)

pat = split_training_test(pat, ptest=0.2)

'''
    INITIALIZE NETWORKS
'''

net = rbm(name='RBM',
          neurogenesis=False,
          nvisible=nvisible,
          nhidden=nhidden-int(np.round(nhidden*0.05)),
          lr=[0.2, 0.2],
          tgt_sparsity=0.05,
          sparsity_cost=[0,0],
          weight_decay=0.3,
          niter=1)

sparse_net = rbm(name='Sparse RBM',
                 neurogenesis=False,
                 nvisible=nvisible,
                 nhidden=nhidden-int(np.round(nhidden*0.05)),
                 lr=[0.2, 0.2],
                 tgt_sparsity=0.05,
                 sparsity_cost=[0.9, 0.9],
                 weight_decay=0.3,
                 niter=1)

net_ng = rbm(name='RBM+NG',
                    neurogenesis=True,
                    nvisible=nvisible,
                    nhidden=nhidden,
                    lr=[0.1, 0.3],
                    tgt_sparsity=0.05,
                    sparsity_cost=[0, 0],
                    weight_decay=0.3,
                    niter=1)

sparse_net_ng = rbm(name='Sparse RBM+NG',
                    neurogenesis=True,
                    nvisible=nvisible,
                    nhidden=nhidden,
                    lr=[0.1, 0.3],
                    tgt_sparsity=0.05,
                    sparsity_cost=[0.1, 0.9],
                    weight_decay=0.3,
                    niter=1)

net.lifespan(pat['pattern']['train'], nblocks=nblocks)
net.test_performance(pat['pattern']['test'])

sparse_net.lifespan(pat['pattern']['train'], nblocks=nblocks)
sparse_net.test_performance(pat['pattern']['test'])

net_ng.lifespan(pat['pattern']['train'], nblocks=nblocks)
net_ng.test_performance(pat['pattern']['test'])

sparse_net_ng.lifespan(pat['pattern']['train'], nblocks=nblocks)
sparse_net_ng.test_performance(pat['pattern']['test'])

'''
    PLOTS
'''

viz.block_reconstruction_error(block_errors=[net.error['blocks'],
                                             sparse_net.error['blocks'],
                                             net_ng.error['blocks'],
                                             sparse_net_ng.error['blocks']],
                               test_errors=[net.error['test'],
                                            sparse_net.error['test'],
                                            net_ng.error['test'],
                                            sparse_net_ng.error['test']],
                               net_names=[net.name,
                                          sparse_net.name,
                                          net_ng.name,
                                          sparse_net_ng.name])


fig, ax = plt.subplots(1, 1)
#ax.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), linestyle='--', c='k', alpha=0.8)
ax.scatter(net.pattern_separation['data'],
           net.pattern_separation['net'],
           c='b',
           alpha='0.1')
ax.scatter(sparse_net.pattern_separation['data'],
           sparse_net.pattern_separation['net'],
           c='r',
           alpha='0.1')
ax.scatter(net_ng.pattern_separation['data'],
           net_ng.pattern_separation['net'],
           c='g',
           alpha='0.1')
ax.scatter(sparse_net_ng.pattern_separation['data'],
           sparse_net_ng.pattern_separation['net'],
           c='m',
           alpha='0.1')
plt.show()
