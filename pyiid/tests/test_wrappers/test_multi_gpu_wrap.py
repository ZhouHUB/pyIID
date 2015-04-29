__author__ = 'christopher'

import numpy as np
from numpy.testing import assert_allclose

from pyiid.wrappers.cpu_wrap import wrap_fq as serial_fq
from pyiid.wrappers.cpu_wrap import wrap_fq_grad as serial_grad_fq

from pyiid.wrappers.multi_gpu_wrap import wrap_fq as gpu_fq
from pyiid.wrappers.multi_gpu_wrap import wrap_fq_grad as gpu_grad_fq

from ase.atoms import Atoms
from pyiid.wrappers.master_wrap import wrap_atoms
n = 40


def test_fq():
    
    pos = np.random.random((n, 3)) * 10.
    atoms = Atoms('Au'+str(n), pos)

    wrap_atoms(atoms)

    sfq_ave = np.zeros(250)
    gfq_ave = np.zeros(250)
    for m in range(10):
        sfq_ave += serial_fq(atoms)
        gfq_ave += gpu_fq(atoms)
    sfq_ave /= m
    gfq_ave /= m
    assert_allclose(sfq_ave, gfq_ave, rtol=1e-3)

    return


def test_grad_fq():
    
    pos = np.random.random((n, 3)) * 10.
    atoms = Atoms('Au'+str(n), pos)

    wrap_atoms(atoms)

    sfq = serial_grad_fq(atoms)
    gfq = gpu_grad_fq(atoms)
    assert_allclose(sfq, gfq, rtol=1e-1)

    return


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)

    import matplotlib.pyplot as plt
    pos = np.random.random((n, 3)) * 10.
    atoms = Atoms('Au'+str(n), pos)

    wrap_atoms(atoms)

    sfq = serial_grad_fq(atoms)
    gfq = gpu_grad_fq(atoms)

    print np.amax(sfq-gfq)
    sfq_ave = sfq[0, 0, :]
    gfq_ave = gfq[0,0,:]

    plt.plot(sfq_ave, label='cpu')
    plt.plot(gfq_ave, label='gpu')
    # plt.plot(np.abs(sfq_ave - gfq_ave)/(gfq_ave+sfq_ave)/2, label='diff')
    plt.plot(np.abs(sfq_ave - gfq_ave), label='diff')
    plt.legend()
    plt.show()