__author__ = 'christopher'
from pyiid.tests import *
from pyiid.wrappers.elasticscatter import ElasticScatter
from pyiid.calc.pdfcalc import PDFCalc

test_data = tuple(product(test_atoms, test_exp, test_potentials,
                              comparison_pro_alg_pairs))

def test_gen_scatter_smoke_fq():
    for v in test_data:
        yield check_scatter_fq, v

def test_gen_scatter_smoke_pdf():
    for v in test_data:
        yield check_scatter_pdf, v

def test_gen_scatter_smoke_sq():
    for v in test_data:
        yield check_scatter_sq, v

def test_gen_scatter_smoke_iq():
    for v in test_data:
        yield check_scatter_iq, v

def test_gen_scatter_smoke_grad_fq():
    for v in test_data:
        yield check_scatter_grad_fq, v

def test_gen_scatter_smoke_grad_pdf():
    for v in test_data:
        yield check_scatter_grad_pdf, v

def check_scatter_fq(value):
    # set everything up
    atoms, exp = value[:2]
    atol = 6e-6 * len(atoms)
    scat = ElasticScatter(exp_dict=exp)
    proc1, alg1 = value[-1][0]
    proc2, alg2 = value[-1][1]

    # run algorithm 1
    scat.set_processor(proc1, alg1)
    ans1 = scat.get_fq(atoms)

    # run algorithm 2
    scat.set_processor(proc2, alg2)
    ans2 = scat.get_fq(atoms)

    # test
    # print np.max(np.abs(ans1 - ans2)), np.mean(
    #     np.abs(ans1 - ans2)), np.std(np.abs(ans1 - ans2))
    assert_allclose(ans1, ans2, atol=atol)
    # assert False

def check_scatter_sq(value):
    # set everything up
    atoms, exp = value[:2]
    atol = 6e-6 * len(atoms)
    scat = ElasticScatter(exp_dict=exp)
    proc1, alg1 = value[-1][0]
    proc2, alg2 = value[-1][1]

    # run algorithm 1
    scat.set_processor(proc1, alg1)
    ans1 = scat.get_sq(atoms)

    # run algorithm 2
    scat.set_processor(proc2, alg2)
    ans2 = scat.get_sq(atoms)

    # test
    assert_allclose(ans1, ans2, rtol=1e-3, atol=atol)

def check_scatter_iq(value):
    # set everything up
    atoms, exp = value[:2]
    atol = 6e-6 * len(atoms)
    scat = ElasticScatter(exp_dict=exp)
    proc1, alg1 = value[-1][0]
    proc2, alg2 = value[-1][1]

    # run algorithm 1
    scat.set_processor(proc1, alg1)
    ans1 = scat.get_iq(atoms)

    # run algorithm 2
    scat.set_processor(proc2, alg2)
    ans2 = scat.get_iq(atoms)

    # test
    assert_allclose(ans1, ans2, rtol=1e-3, atol=atol)

def check_scatter_pdf(value):
    # set everything up
    atoms, exp = value[:2]
    atol = 6e-6 * len(atoms)
    scat = ElasticScatter(exp_dict=exp)
    proc1, alg1 = value[-1][0]
    proc2, alg2 = value[-1][1]

    # run algorithm 1
    scat.set_processor(proc1, alg1)
    ans1 = scat.get_pdf(atoms)

    # run algorithm 2
    scat.set_processor(proc2, alg2)
    ans2 = scat.get_pdf(atoms)

    # test
    assert_allclose(ans1, ans2, atol=atol)

def check_scatter_grad_fq(value):
    # set everything up
    atoms, exp = value[:2]
    atol = 6e-6 * len(atoms)
    scat = ElasticScatter(exp_dict=exp)
    proc1, alg1 = value[-1][0]
    proc2, alg2 = value[-1][1]

    # run algorithm 1
    scat.set_processor(proc1, alg1)
    ans1 = scat.get_grad_fq(atoms)

    # run algorithm 2
    scat.set_processor(proc2, alg2)
    ans2 = scat.get_grad_fq(atoms)

    # test
    assert_allclose(ans1, ans2, atol=atol)

def check_scatter_grad_pdf(value):
    # set everything up
    atoms, exp = value[:2]
    atol = 6e-6 * len(atoms)
    scat = ElasticScatter(exp_dict=exp)
    proc1, alg1 = value[-1][0]
    proc2, alg2 = value[-1][1]

    # run algorithm 1
    scat.set_processor(proc1, alg1)
    ans1 = scat.get_grad_pdf(atoms)

    # run algorithm 2
    scat.set_processor(proc2, alg2)
    ans2 = scat.get_grad_pdf(atoms)

    # test
    assert_allclose(ans1, ans2, atol=atol)
# '''
if __name__ == '__main__':
    import nose

    nose.runmodule(argv=[
        # '-s',
        '--with-doctest',
        '--nocapture',
        # '-v'
    ],
        # env={"NOSE_PROCESSES": 1, "NOSE_PROCESS_TIMEOUT": 599},
        exit=False)
