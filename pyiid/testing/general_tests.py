__author__ = 'christopher'
from pyiid.tests import *
from pyiid.wrappers.elasticscatter import ElasticScatter
from pyiid.calc.pdfcalc import PDFCalc
from pympler import muppy, summary
"""
These are non-unique tests, they are run for a combination of processors,
algorithms, experiments, and atomic configurations.  The number of tests
can be modified in the test init file.  The purpose for the large number of
potentially expensive tests is to make certain that the code runs over a large
number of potential configurations and get full code coverage.  Some of the
good side effects of this extensive testing is that we find issues associated
with memory leaks, and issues of how we break up tasks over multiprocessors.
"""
'''
def tearDownModule():
    print 'final'
    all_objects = muppy.get_objects()
    sum1 = summary.summarize(all_objects)
    summary.print_(sum1)

@classmethod
    def tearDownClass(cls):
        all_objects = muppy.get_objects()
        sum1 = summary.summarize(all_objects)
        summary.print_(sum1)
        del cls.test_data
'''
# single proc/alg to smoke test
@ddt
class TestScatterSmoke(TC):
    """
    Test flat cpu scatter
    """
    test_data = tuple(product(test_atoms, test_exp, proc_alg_pairs))

    @data(*test_data)
    def test_scatter_fq(self, value):
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

    @data(*test_data)
    def test_scatter_pdf(self, value):
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

    @data(*test_data)
    def test_scatter_grad_fq(self, value):
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

    @data(*test_data)
    def test_scatter_grad_pdf(self, value):
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

@ddt
class TestPDFCalcKnown(TC):
    """
    Test flat cpu scatter
    """
    test_data = tuple(
        product(test_atom_squares, test_exp, test_potentials, proc_alg_pairs))

    @data(*test_data)
    def test_nrg(self, value):
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

    @data(*test_data)
    def test_forces(self, value):
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


# double proc/alg for comparison between kernels
@ddt
class TestPDFCalc(TC):
    """
    Test flat cpu scatter
    """
    test_data = tuple(product(test_double_atoms, test_exp, test_potentials,
                              comparison_pro_alg_pairs))

    @data(*test_data)
    def test_nrg(self, value):
        # setup
        atoms1, atoms2 = value[0]
        scat = ElasticScatter()
        proc1, alg1 = value[-1][0]
        proc2, alg2 = value[-1][1]
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
        # print np.max(np.abs(ans2 - ans1)), np.mean(
        #     np.abs(ans2 - ans1)), np.std(np.abs(ans2 - ans1))
        assert_allclose(ans2, ans1,
                        # rtol=5e-4,
                        atol=1e-3)

    @data(*test_data)
    def test_forces(self, value):
        # setup
        atoms1, atoms2 = value[0]
        scat = ElasticScatter()
        proc1, alg1 = value[-1][0]
        proc2, alg2 = value[-1][1]
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
        # print np.max(np.abs(ans2 - ans1)), np.mean(
        #     np.abs(ans2 - ans1)), np.std(np.abs(ans2 - ans1))
        assert_allclose(ans2, ans1,
                        # rtol=5e-4,
                        atol=1e-3)

@ddt
class TestScatter(TC):
    """
    Test flat cpu scatter
    """

    test_data = tuple(product(test_atoms, test_exp, test_potentials,
                              comparison_pro_alg_pairs))

    @data(*test_data)
    def test_scatter_fq(self, value):
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

    @data(*test_data)
    def test_scatter_sq(self, value):
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

    @data(*test_data)
    def test_scatter_iq(self, value):
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

    @data(*test_data)
    def test_scatter_pdf(self, value):
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

    @data(*test_data)
    def test_scatter_grad_fq(self, value):
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

    @data(*test_data)
    def test_scatter_grad_pdf(self, value):
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
