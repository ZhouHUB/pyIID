__author__ = 'christopher'
from pyiid.tests import *
from pyiid.experiments.elasticscatter import ElasticScatter

local_test_atoms = setup_atomic_square()[0] * 3
test_data = tuple(product([local_test_atoms], [None]))

def test_fq_against_srfit():
    for value in test_data:
        yield check_fq_against_srfit, value

@known_fail_if(not srfit)
def check_fq_against_srfit(value):
    # unpack the atoms and experiment
    atoms = value[0]
    exp = value[1]

    # get the pyIID F(Q)
    s = ElasticScatter(exp)
    # s.set_processor('CPU', 'nxn')
    ans1 = s.get_fq(atoms)

    # get the SrFit F(Q)
    stru = convert_atoms_to_stru(atoms)
    srfit_calc = DebyePDFCalculator()
    srfit_calc.qmin = s.exp['qmin']
    srfit_calc.qmax = s.exp['qmax']
    srfit_calc.qstep = s.exp['qbin']
    r1, g1 = srfit_calc(stru)
    assert_allclose(s.get_scatter_vector(), srfit_calc.qgrid)
    ans2 = srfit_calc.fq
    stats_check(ans1, ans2, rtol=1e-1, atol=5e-6)
    del srfit_calc
    assert_allclose(ans1, ans2, rtol=1e-1, atol=5e-6)

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
