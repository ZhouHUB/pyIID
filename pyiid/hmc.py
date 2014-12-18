__author__ = 'christopher'
import numpy as np
from ase.atoms import Atoms
from copy import deepcopy as dc
atoms = Atoms


def leapfrog(atoms, step):
    grad = atoms.get_forces()
    atoms.set_momenta(atoms.get_momenta() - step*grad)
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
    return atoms.get_potential_energy() + kin_energy(atoms)


def hmc_move(atoms, stepsize, n_steps):
    atoms.set_momenta(np.random.normal(0, .1, (len(atoms), 3)))
    atoms2 = dc(atoms)
    simulate_dynamics(atoms, stepsize, n_steps)
    accept = mh_accept(hamil(atoms), hamil(atoms2))
    if accept is True:
        return accept, atoms2
    else:
        return accept, atoms

def run_hmc(atoms, iterations, stepsize, n_steps, avg_acceptance_slowness,
            avg_acceptance_rate, target_acceptance_rate, stepsize_inc,
            stepsize_dec):
    traj = [atoms]
    accept_list = []
    initial_energy = atoms.get_potential_energy()
    i = 0
    while i < iterations:
        accept, atoms = hmc_move(atoms, stepsize, n_steps)
        if accept is True:
            traj += atoms
        accept_list.append(accept)
        avg_acceptance_rate = avg_acceptance_slowness * avg_acceptance_rate + (1.0 - avg_acceptance_slowness) * np.average(accept_list)
        if avg_acceptance_rate > target_acceptance_rate :
            stepsize *= stepsize_inc
        else:
            stepsize *= stepsize_dec

if __name__ == '__main__':
    # import cProfile
    from ase import Atoms
    import matplotlib.pyplot as plt
    # cProfile.run('''
    import ase.io as aseio
    from copy import deepcopy as dc
    # atoms = aseio.read('/home/christopher/pdfgui_np_25_rattle1_cut.xyz')
    atoms = Atoms('Au4', [(0,0,0), (3,0,0), (0,3,0), (3,3,0)])
    pdf, FQ = wrap_pdf(atoms, Qmin=2.5)
    calc = pdf_calc(gobs=pdf, Qmin=2.5)
    atoms.set_calculator(calc)

    # plt.show()
    atoms2 = dc(atoms)
    atoms2.rattle(stdev=.1)
    pdf2, FQ2 = wrap_pdf(atoms2, Qmin=2.5)
    atoms2.set_calculator(calc)
    # pdf2, FQ2 = wrap_pdf(atoms2)
    print 'start energy'
    # t0 = time.time()
    print(atoms2.get_total_energy())
    # t1 = time.time()
    # print(atoms2.get_forces())
    # t2 = time.time()
    # ''', sort='tottime')
    #     print('energy', t1-t0, 'forces', t2-t1)
    plt.plot(pdf)
    plt.plot(pdf2)
    # plt.plot(pdf-pdf2)
    plt.show()