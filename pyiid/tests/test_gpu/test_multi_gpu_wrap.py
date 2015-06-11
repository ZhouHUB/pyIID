__author__ = 'christopher'

import numpy as np
from numpy.testing import assert_allclose
from ase.atoms import Atoms

from pyiid.wrappers.cpu_wrap import wrap_fq as serial_fq
from pyiid.wrappers.cpu_wrap import wrap_fq_grad as serial_grad_fq

from pyiid.wrappers.multi_gpu_wrap import wrap_fq as gpu_fq
from pyiid.wrappers.multi_gpu_wrap import wrap_fq_grad as gpu_grad_fq

from pyiid.wrappers.elasticscatter import wrap_atoms
from pyiid.tests import generate_experiment

n = 150


def test_fq():
    
    pos = np.random.random((n, 3)) * 10.
    atoms = Atoms('Au'+str(n), pos)

    exp_dict = generate_experiment()
    wrap_atoms(atoms, exp_dict)

    gfq = gpu_fq(atoms, exp_dict['qbin'])
    sfq = serial_fq(atoms, exp_dict['qbin'])
    print 'rms', np.sqrt(np.mean((gfq - sfq) ** 2))
    print 'mean difference', np.mean(gfq - sfq)
    print 'median difference', np.median(gfq - sfq)
    assert_allclose(sfq, gfq, rtol=1e-2, atol=.0000001)

    return


def test_grad_fq():
    
    pos = np.random.random((n, 3)) * 10.
    atoms = Atoms('Au'+str(n), pos)

    exp_dict = generate_experiment()
    wrap_atoms(atoms, exp_dict)

    gfq = gpu_grad_fq(atoms, exp_dict['qbin'])
    sfq = serial_grad_fq(atoms, exp_dict['qbin'])
    print 'rms', np.sqrt(np.mean((gfq - sfq) ** 2))
    print 'mean difference', np.mean(gfq - sfq)
    print 'median difference', np.median(gfq - sfq)
    assert_allclose(sfq, gfq, rtol=1e-1, atol=.0000001)

    return


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest', '-v'], exit=False)