__author__ = 'christopher'
from pyiid.tests import *
from pyiid.wrappers.elasticscatter import ElasticScatter

dq = 1e-4


def finite_difference_grad(atoms, exp_dict):
    s = ElasticScatter(exp_dict)
    start_fq = s.get_fq(atoms)
    finite_difference_grad_fq = np.zeros((len(atoms), 3, len(start_fq)))
    for i in range(len(atoms)):
        for w in range(3):
            atoms2 = dc(atoms)
            atoms2[i].position[w] += dq
            fq2 = s.get_fq(atoms2)
            finite_difference_grad_fq[i, w, :] = -.5 * (fq2 - start_fq) / dq
    return finite_difference_grad_fq


if __name__ == '__main__':
    rt = 1e-5
    at=2e-2
    import matplotlib.pyplot as plt
    atoms = setup_atomic_square()[0]
    # atoms = Atoms('Au2', [[0, 0, 0], [1, 0, 0]])
    exp = None
    s = ElasticScatter(exp)
    a = finite_difference_grad(atoms, exp)
    b = s.get_grad_fq(atoms)
    # plt.plot(a[0, 0, :], label='fd')
    # plt.plot(b[0, 0, :], label='analytical')
    # plt.legend()
    # plt.show()
    stats_check(a, b, rt, at)
    assert_allclose(a, b, rtol=rt, atol=at)
