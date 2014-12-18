__author__ = 'christopher'
import numpy as np
from ase.atoms import Atoms
from copy import deepcopy as dc
atoms = Atoms


def leapfrog(atoms, step):
    grad = atoms.get_forces()
    atoms.set_momenta(atoms.get_momenta() + step*grad)
    atoms.set_positions(atoms.get_positions() + step*atoms.get_momenta())


def simulate_dynamics(atoms, stepsize, n_steps):
    #get the initial positions and momenta
    grad = atoms.get_forces()
    atoms.set_momenta(atoms.get_momenta() - 0.5 * stepsize * grad)
    atoms.set_positions(atoms.get_positions() + stepsize * atoms.get_momenta())
    for n in range(n_steps):
        leapfrog(atoms, stepsize)
    atoms.set_momenta(atoms.get_momenta() - 0.5 * stepsize * atoms.get_forces())


def mh_accept(inital_energy, next_energy):
    diff = inital_energy - next_energy
    return (np.exp(diff) - np.random.random(inital_energy.shape)) >= 0


def kin_energy(atoms):
    return np.sum(atoms.get_momenta() ** 2) / 2.0

def hamil(atoms):
    print 'hamil'
    print atoms.get_potential_energy()
    print kin_energy(atoms)
    return atoms.get_potential_energy() + kin_energy(atoms)


def hmc_move(atoms, stepsize, n_steps):
    prop = dc(atoms)
    prop.set_momenta(np.random.normal(0, .1, (len(atoms), 3)))
    simulate_dynamics(prop, stepsize, n_steps)
    accept = mh_accept(hamil(atoms), hamil(prop))
    # if accept is True:
    #     return accept, atoms2
    # else:
    #     return accept, atoms
    return accept, prop
def run_hmc(atoms, iterations, stepsize, n_steps, avg_acceptance_slowness,
            avg_acceptance_rate, target_acceptance_rate, stepsize_inc,
            stepsize_dec, stepsize_min, stepsize_max):
    traj = [atoms]
    accept_list = []
    initial_energy = atoms.get_potential_energy()
    i = 0
    while i < iterations:
        accept, atoms = hmc_move(atoms, stepsize, n_steps)
        print i, accept, atoms.get_potential_energy(),stepsize
        # if accept is True:
        traj += atoms
        accept_list.append(accept)
        avg_acceptance_rate = avg_acceptance_slowness * avg_acceptance_rate + (1.0 - avg_acceptance_slowness) * np.average(np.array(accept_list))
        if avg_acceptance_rate > target_acceptance_rate :
            stepsize *= stepsize_inc
        else:
            stepsize *= stepsize_dec
        if stepsize > stepsize_max:
            stepsize = stepsize_max
        elif stepsize < stepsize_min:
            stepsize = stepsize_min

        # print i
        i += 1
    return traj, accept

if __name__ == '__main__':
    # import cProfile
    from ase import Atoms
    from pyiid.kernel_wrap import wrap_rw, wrap_grad_rw, wrap_pdf
    import matplotlib.pyplot as plt
    from pyiid.pdf_calc import pdf_calc
    # cProfile.run('''
    import ase.io as aseio
    from copy import deepcopy as dc
    # atoms = aseio.read('/home/christopher/pdfgui_np_25_rattle1_cut.xyz')
    atoms = Atoms('Au4', [(0,0,0), (3,0,0), (0,3,0), (3,3,0)])
    pdf, FQ = wrap_pdf(atoms,
                       # Qmin=2.5
    )
    calc = pdf_calc(gobs=pdf,
                    conv=.0001
                    # Qmin=2.5
    )
    atoms.set_calculator(calc)

    # plt.show()
    atoms2 = dc(atoms)
    atoms2.rattle(stdev=.1)
    atoms2.set_calculator(calc)
    traj, accept = run_hmc(atoms2, 10, .00001, 5, 0.9,
            0, .9, 1.02,
            .98, .001, .25)
    # ''', sort='tottime')
    aseio.write('/home/christopher/dev/pyIID/extra/au_test.traj', traj)
    # plt.plot(pdf)
    # plt.plot(pdf2)
    # plt.plot(pdf-pdf2)
    # plt.show()