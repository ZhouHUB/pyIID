from pyiid.tests import *
from pyiid.experiments.elasticscatter import ElasticScatter
from pyiid.calc.pdfcalc import PDFCalc
__author__ = 'christopher'

test_data = tuple(
    product(test_atom_squares, test_exp, test_potentials, proc_alg_pairs))


def test_nrg():
    for v in test_data:
        yield check_nrg, v


def test_forces():
    for v in test_data:
        yield check_forces, v


def check_nrg(value):
    """
    Check for PDF energy against known value
    :param value:
    :return:
    """
    # setup
    atoms1, atoms2 = value[0]
    proc, alg = value[-1]
    p, thresh = value[2]
    scat = ElasticScatter()
    scat.update_experiment(exp_dict=value[1])
    scat.set_processor(proc, alg)

    gobs = scat.get_pdf(atoms1)
    calc = PDFCalc(obs_data=gobs, scatter=scat, potential=p)
    atoms2.set_calculator(calc)

    ans = atoms2.get_potential_energy()
    assert ans >= thresh
    del atoms1, atoms2, proc, alg, p, thresh, scat, gobs, calc, ans


def check_forces(value):
    """
    Check for PDF forces against known value
    :param value:
    :return:
    """
    # setup
    atoms1, atoms2 = value[0]
    proc, alg = value[-1]
    p, thresh = value[2]
    scat = ElasticScatter()
    scat.update_experiment(exp_dict=value[1])
    scat.set_processor(proc, alg)

    gobs = scat.get_pdf(atoms1)
    calc = PDFCalc(obs_data=gobs, scatter=scat, potential=p)
    atoms2.set_calculator(calc)

    forces = atoms2.get_forces()
    com = atoms2.get_center_of_mass()
    for i in range(len(atoms2)):
        dist = atoms2[i].position - com
        # print i, dist, forces[i], np.cross(dist, forces[i])
        assert_allclose(np.cross(dist, forces[i]), np.zeros(3))
    del atoms1, atoms2, proc, alg, p, thresh, scat, gobs, calc, forces, com, dist


if __name__ == '__main__':
    import nose

    nose.runmodule(argv=[
        # '-s',
        '--with-doctest',
        # '--nocapture',
        # '-v'
    ],
        # env={"NOSE_PROCESSES": 1, "NOSE_PROCESS_TIMEOUT": 599},
        exit=False)
