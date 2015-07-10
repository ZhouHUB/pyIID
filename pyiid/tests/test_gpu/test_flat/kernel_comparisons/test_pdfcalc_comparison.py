__author__ = 'christopher'
from pyiid.wrappers.elasticscatter import ElasticScatter
from pyiid.calc.pdfcalc import PDFCalc
from pyiid.tests import *

test_data = tuple(product(test_double_atoms, test_exp, test_potentials))
proc1 = 'CPU'
proc2 = 'Multi-GPU'
alg1 = 'flat'
alg2 = 'flat'

@ddt
class TestPDFCalc(TC):
    """
    Test flat cpu scatter
    """
    @data(*test_data)
    def test_nrg(self, value):
        # setup
        atoms1, atoms2 = value[0]
        scat = ElasticScatter()
        scat.update_experiment(exp_dict=value[1])
        scat.set_processor(proc1, alg1)
        p, thresh = value[2]
        gobs = scat.get_pdf(atoms1)

        calc = PDFCalc(obs_data=gobs, scatter=scat, potential=p)
        atoms2.set_calculator(calc)
        ans1 = atoms2.get_potential_energy()
        print scat.processor, scat.alg

        scat.set_processor(proc2, alg2)
        calc = PDFCalc(obs_data=gobs, scatter=scat, potential=p)
        atoms2.set_calculator(calc)
        ans2 = atoms2.get_potential_energy()

        print scat.processor, scat.alg
        print np.max(np.abs(ans2 - ans1)), np.mean(np.abs(ans2 - ans1)), np.std(np.abs(ans2 - ans1))
        assert_allclose(ans2, ans1,
                        # rtol=5e-4,
                        atol=1e-3)

    @data(*test_data)
    def test_forces(self, value):
        # setup
        atoms1, atoms2 = value[0]
        scat = ElasticScatter()
        scat.update_experiment(exp_dict=value[1])
        scat.set_processor(proc1, alg1)
        p, thresh = value[2]
        gobs = scat.get_pdf(atoms1)

        calc = PDFCalc(obs_data=gobs, scatter=scat, potential=p)
        atoms2.set_calculator(calc)
        ans1 = atoms2.get_forces()
        print scat.processor, scat.alg

        scat.set_processor(proc2, alg2)
        calc = PDFCalc(obs_data=gobs, scatter=scat, potential=p)
        atoms2.set_calculator(calc)
        ans2 = atoms2.get_forces()

        print np.max(np.abs(ans2 - ans1)), np.mean(np.abs(ans2 - ans1)), np.std(np.abs(ans2 - ans1))
        print scat.processor, scat.alg
        assert_allclose(ans2, ans1,
                        # rtol=5e-4,
                        atol=1e-3)

if __name__ == '__main__':
    import nose

    nose.runmodule(argv=['-s', '--with-doctest', '-v'], exit=False)
