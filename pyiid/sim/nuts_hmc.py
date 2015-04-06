__author__ = 'christopher'
from numbapro import autojit

from ase.atoms import Atoms
from copy import deepcopy as dc
import numpy as np

atoms = Atoms

@autojit(target='cpu')
def leapfrog(atoms, step):
    """
    Propagate the dynamics of the system via the leapfrog algorithm one step

    Parameters
    -----------
    atoms: ase.Atoms ase.Atoms
        The atomic configuration for the system
    step: float
        The step size for the simulation, the new momentum/velocity is step *
        the force

    Returns
    -------
    ase.Atoms
        The new atomic positions and velocities
    """
    atoms.set_momenta(atoms.get_momenta() + 0.5 * step * atoms.get_forces())


    # atoms.set_positions(atoms.get_positions() + step * atoms.get_velocities())

    atoms.positions += step * atoms.get_velocities()

    atoms.set_momenta(atoms.get_momenta() + 0.5 * step * atoms.get_forces())
    return atoms


def find_step_size(atoms):
    step_size = 1.
    # '''
    atoms.set_momenta(
        np.random.normal(0, 1, (len(atoms), 3)))
    old_atoms = dc(atoms)
    h_old = old_atoms.get_total_energy()
    leapfrog(atoms, step_size)
    a = 2 * (np.exp(-atoms.get_total_energy()+h_old) > 0.5) - 1
    while (np.exp(-atoms.get_total_energy() + h_old)) ** a > 2 ** -a:
        step_size *= 2 ** a
        leapfrog(atoms, step_size)
        # print np.exp(-atoms.get_total_energy() + h_old)
        # '''
    return step_size


def bern(p):
    return np.random.uniform() < p
    # return np.random.random() < p


def nuts_da_hmc(atoms, accept_target, iterations, Emax):
    start_atoms = dc(atoms)
    traj = [start_atoms]
    step_size = find_step_size(atoms)
    atoms = dc(start_atoms)
    m = 0
    t0 = 10
    Hbar = 0
    gamma = 0.05
    mu = np.log(10*step_size)

    try:
        while m <= iterations:
            e = step_size

            u = np.random.uniform(0, np.exp(-atoms.get_total_energy()))
            # u = np.random.random()
            n, s, j = 1, 1, 0
            atoms0 = dc(atoms)
            atomsn = dc(atoms)
            atomsp = dc(atoms)
            while s == 1:
                v = bern(.5) * 2 - 1

                if v == -1:
                    atomsn, _, atoms1, n1, s1, a, na = \
                        buildtree(atomsn, u, v, j, e, Emax, atoms0)
                else:
                    _, atomsp, atoms1, n1, s1, a, na = \
                        buildtree(atomsp, u, v, j, e, Emax, atoms0)

                if s1 == 1 and bern(min(1, n1 * 1. / n)):
                    print 'accepted', atoms1.get_potential_energy()
                    atoms = atoms1
                    traj += [atoms]

                # if atoms1.get_potential_energy() < atoms.get_potential_energy():
                #     atoms = atoms1

                n = n + n1

                span = atomsp.positions - atomsn.positions
                span = span.flatten()
                # print 'emax criteria', s1
                # print 'negative u turn', span.dot(atomsn.get_velocities().flatten()) >= 0
                # print 'positive u turn', span.dot( atomsp.get_velocities().flatten()) >= 0
                # print 'mh critera', min(1, n1 * 1. / n)
                # print 'n prime', n1, 'n', n

                s = s1 * \
                    (span.dot(atomsn.get_velocities().flatten()) >= 0) * \
                    (span.dot(atomsp.get_velocities().flatten()) >= 0)
                j += 1

            print 'm', m, 'j', j, 'e', e, 'nrg', atoms1.get_potential_energy()

            w = 1. / (m + t0)
            Hbar = (1 - w) * Hbar + w * (accept_target - a * 1. / na)

            step_size = np.exp(mu - (m ** .5 / gamma) * Hbar)

            m += 1
            if atoms != traj[-1]:
            #     traj += [atoms]
                print 'accepted', atoms.get_potential_energy()
            print '-------------'
    except KeyboardInterrupt:
        pass
    return traj

@autojit(target='cpu')
def buildtree(atoms, u, v, j, e, Emax, atoms0):
    if j == 0:
        atoms1 = leapfrog(atoms, v*e)
        E = atoms1.get_total_energy()
        E0 = atoms0.get_total_energy()
        # E = atoms1.get_potential_energy()
        # E0 = atoms0.get_potential_energy()

        dE = E - E0

        n1 = int(np.log(u) + dE <= 0)
        s1 = int(np.log(u) + dE < Emax)
        # print dE
        return atoms1, atoms1, atoms1, n1, s1, min(1, np.exp(-dE)), 1
    else:
        atomsn, atomsp, atoms1, n1, s1, a1, na1 = \
            buildtree(atoms, u, v, j - 1, e, Emax, atoms0)
        if s1 == 1:
            if v == -1:
                atomsn, _, atoms11, n11, s11, a11, na11 = \
                    buildtree(atomsn, u, v, j - 1, e, Emax, atoms0)
            else:
                _, atomsp, atoms11, n11, s11, a11, na11 = \
                    buildtree(atomsp, u, v, j - 1, e, Emax, atoms0)

            if bern(n11 * 1. / (max(n1 + n11, 1))):
                atoms1 = atoms11

            a1 = a1 + a11
            na1 = na1 + na11

            span = atomsp.positions - atomsn.positions
            span = span.flatten()
            s1 = s11 * (span.dot(atomsn.get_velocities().flatten()) >= 0) * (
                span.dot(atomsp.get_velocities().flatten()) >= 0)
            n1 = n1 + n11
        return atomsn, atomsp, atoms1, n1, s1, a1, na1


if __name__ == '__main__':
    import os
    from copy import deepcopy as dc
    import numpy as np
    import matplotlib.pyplot as plt
    from ase.visualize import view
    from ase.io.trajectory import PickleTrajectory
    import ase.io as aseio
    import time
    import datetime
    import math
    from ase.atoms import Atoms

    from pyiid.sim.hmc import run_hmc
    from pyiid.wrappers.multi_gpu_wrap import wrap_rw, wrap_pdf
    from pyiid.calc.pdfcalc import PDFCalc
    from pyiid.wrappers.kernel_wrap import wrap_atoms
    from pyiid.utils import tag_surface_atoms, build_sphere_np, load_gr_file, time_est

    ideal_atoms = Atoms('Au4', [[0, 0, 0], [3, 0, 0], [0, 3, 0], [3, 3, 0]])
    # start_atoms = Atoms('Au4', [[-1, 0, 0], [3, 0, 0], [0, 3, 0], [3, 3, 0]])
    start_atoms = dc(ideal_atoms)
    start_atoms.positions *= 1.05

    wrap_atoms(ideal_atoms)
    wrap_atoms(start_atoms)

    gobs, fq = wrap_pdf(ideal_atoms, qbin=.1)

    calc = PDFCalc(gobs=gobs, conv=100, potential='rw', qbin=.1)
    # calc = PDFCalc(gobs=gobs, conv=1000, qbin=.1)
    start_atoms.set_calculator(calc)
    print start_atoms.get_potential_energy()

    e = .8e-2
    M = 10

    traj, C_list = nuts(start_atoms, e, M)
    for c in C_list:
        for atoms in c:
            f = atoms.get_forces()
            f2 = f*2
        view(c)

    # raw_input('')
    pe_list = []
    for atoms in traj:
        pe_list.append(atoms.get_potential_energy())
        f = atoms.get_forces()
        f2 = f*2
    min_pe = np.argmin(pe_list)
    view(traj)
    rw, scale, _, _ = wrap_rw(traj[-1], gobs)
    print rw, scale
    r = np.arange(0, 40, .01)
    plt.plot(r, gobs, label='gobs')
    plt.plot(r, wrap_pdf(traj[0], qbin=.1)[0]*wrap_rw(traj[0],gobs, qbin=.1)[1], label='ginitial')
    plt.plot(r, wrap_pdf(traj[min_pe], qbin=.1)[0]*wrap_rw(traj[min_pe],gobs, qbin=.1)[1], label='gfinal')
    plt.legend()
    plt.show()