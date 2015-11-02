from pyiid.tests import *
from pyiid.experiments.elasticscatter import ElasticScatter
from time import time

__author__ = 'christopher'

def test_exp():
    atoms = setup_atoms(int(10))
    exp1 = generate_experiment()
    exp2 = generate_experiment()

    scat = ElasticScatter(exp_dict=exp1)
    assert scat.check_state(atoms)
    fq = scat.get_fq(atoms)
    gfq = scat.get_grad_fq(atoms)
    assert not scat.check_state(atoms)
    assert fq is not None

    ff = atoms.arrays['F(Q) scatter']

    scat.update_experiment(exp_dict=exp2)
    fq2 = scat.get_fq(atoms)
    gfq2 = scat.get_grad_fq(atoms)
    ff2 = atoms.arrays['F(Q) scatter']

    if ff.shape == ff2.shape:
        assert not np.allclose(ff, ff2)
    if fq.shape == fq2.shape:
        assert not np.allclose(fq, fq2)
    if gfq.shape == gfq2.shape:
        assert not np.allclose(gfq, gfq2)


def test_number_of_atoms():
    atoms = setup_atoms(int(10))
    atoms2 = setup_atoms(int(11))
    exp1 = generate_experiment()

    scat = ElasticScatter(exp_dict=exp1)
    assert scat.check_state(atoms)
    fq = scat.get_fq(atoms)
    assert not scat.check_state(atoms)
    assert fq is not None

    fq2 = scat.get_fq(atoms2)
    assert not np.allclose(fq, fq2)

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[
        # '-s',
        '--with-doctest',
        # '--nocapture',
        '-v',
        # '-x',
    ],
        # env={"NOSE_PROCESSES": 1, "NOSE_PROCESS_TIMEOUT": 599},
        exit=False)
