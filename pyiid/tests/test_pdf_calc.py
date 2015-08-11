__author__ = 'christopher'
from pyiid.tests import *
from pyiid.wrappers.elasticscatter import ElasticScatter
from pyiid.calc.pdfcalc import PDFCalc

rtol = 1e-5
atol = 1e-5

test_data = tuple(product(test_double_atoms, test_exp, test_potentials,
                          comparison_pro_alg_pairs, [rtol], [atol]))


def test_forces():
    for v in test_data:
        yield check_forces, v


def test_nrg():
    for v in test_data:
        yield check_nrg, v


def check_nrg(value):
    """
    Check two processor, algorithm pairs against each other for PDF energy
    :param value:
    :return:
    """
    # setup
    atoms1, atoms2 = value[0]
    scat = ElasticScatter()
    proc1, alg1 = value[3][0]
    proc2, alg2 = value[3][1]
    scat.update_experiment(exp_dict=value[1])
    scat.set_processor(proc1, alg1)
    p, thresh = value[2]
    gobs = scat.get_pdf(atoms1)

    calc = PDFCalc(obs_data=gobs, scatter=scat, potential=p)
    atoms2.set_calculator(calc)
    ans1 = atoms2.get_potential_energy()

    scat.set_processor(proc2, alg2)
    calc = PDFCalc(obs_data=gobs, scatter=scat, potential=p)
    atoms2.set_calculator(calc)
    ans2 = atoms2.get_potential_energy()
    print ans2, ans1, np.abs(ans1 - ans2), atol+rtol*np.abs(ans1)
    assert_allclose(ans2, ans1, rtol=rtol, atol=atol)


def check_forces(value):
    """
    Check two processor, algorithm pairs against each other for PDF forces
    :param value:
    :return:
    """
    # setup
    rtol = value[4]
    atol = value[5]
    atoms1, atoms2 = value[0]
    scat = ElasticScatter()
    proc1, alg1 = value[3][0]
    proc2, alg2 = value[3][1]
    scat.update_experiment(exp_dict=value[1])
    scat.set_processor(proc1, alg1)
    p, thresh = value[2]
    gobs = scat.get_pdf(atoms1)

    calc = PDFCalc(obs_data=gobs, scatter=scat, potential=p)
    atoms2.set_calculator(calc)
    ans1 = atoms2.get_forces()

    scat.set_processor(proc2, alg2)
    calc = PDFCalc(obs_data=gobs, scatter=scat, potential=p)
    atoms2.set_calculator(calc)
    ans2 = atoms2.get_forces()
    if p == 'chi_sq':
        rtol *= 20
        atol *= 20
    stats_check(ans1, ans2, rtol, atol)
    assert_allclose(ans2, ans1, rtol=rtol, atol=atol)

if __name__ == '__main__':
    import nose

    nose.runmodule(argv=[
        # '-s',
        '--with-doctest',
        # '--nocapture',
        # '-v'
        '-x'
    ],
        # env={"NOSE_PROCESSES": 1, "NOSE_PROCESS_TIMEOUT": 599},
        exit=False)
