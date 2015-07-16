__author__ = 'christopher'
from pyiid.tests import *
from pyiid.wrappers.elasticscatter import ElasticScatter
from pyiid.calc.pdfcalc import PDFCalc

test_data = tuple(product(test_atoms, test_exp, proc_alg_pairs))


def test_gen_scatter_smoke_fq():
    for v in test_data:
        yield check_scatter_fq, v

def test_gen_scatter_smoke_pdf():
    for v in test_data:
        yield check_scatter_pdf, v

def test_gen_scatter_smoke_grad_fq():
    for v in test_data:
        yield check_scatter_grad_fq, v

def test_gen_scatter_smoke_grad_pdf():
    for v in test_data:
        yield check_scatter_grad_pdf, v

def check_scatter_fq(value):
    atoms, exp = value[0:2]
    proc, alg = value[-1]

    scat = ElasticScatter(exp_dict=exp)
    scat.set_processor(proc, alg)
    # Test a set of different sized ensembles
    ans = scat.get_fq(atoms)
    # Check that Scatter gave back something
    assert ans is not None
    # Check that all the values are not zero
    assert np.any(ans)
    del atoms, exp, proc, alg, scat, ans
    return

def check_scatter_pdf(value):
    atoms, exp = value[0:2]
    proc, alg = value[-1]

    scat = ElasticScatter(exp_dict=exp)
    scat.set_processor(proc, alg)
    # Test a set of different sized ensembles
    ans = scat.get_pdf(atoms)
    # Check that Scatter gave back something
    assert ans is not None
    # Check that all the values are not zero
    assert np.any(ans)
    del atoms, exp, proc, alg, scat, ans
    return

def check_scatter_grad_fq(value):
    atoms, exp = value[0:2]
    proc, alg = value[-1]

    scat = ElasticScatter(exp_dict=exp)
    scat.set_processor(proc, alg)
    # Test a set of different sized ensembles
    ans = scat.get_grad_fq(atoms)
    # Check that Scatter gave back something
    assert ans is not None
    # Check that all the values are not zero
    assert np.any(ans)
    del atoms, exp, proc, alg, scat, ans
    return

def check_scatter_grad_pdf(value):
    atoms, exp = value[0:2]
    proc, alg = value[-1]

    scat = ElasticScatter(exp_dict=exp)
    scat.set_processor(proc, alg)
    # Test a set of different sized ensembles
    ans = scat.get_grad_pdf(atoms)
    # Check that Scatter gave back something
    assert ans is not None
    # Check that all the values are not zero
    assert np.any(ans)
    del atoms, exp, proc, alg, scat, ans
    return


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
