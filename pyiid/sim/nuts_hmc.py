from copy import deepcopy as dc
from ase.units import fs
import numpy as np
from numpy.random import RandomState

from pyiid.sim import leapfrog

__author__ = 'christopher'
Emax = 200


def find_step_size(input_atoms, rs):
    """
    Find a suitable starting step size for the simulation

    Parameters
    -----------
    input_atoms: ase.Atoms object
        The starting atoms for the simulation
    rs: numpy.random.RandomState object
        The random number generator for this simulation
    Returns
    -------
    float:
        The step size
    """
    atoms = dc(input_atoms)
    step_size = .5
    atoms.set_momenta(rs.normal(0, 1, (len(atoms), 3)))

    atoms_prime = leapfrog(atoms, step_size)

    a = 2 * (np.exp(
        -1 * atoms_prime.get_total_energy() + atoms.get_total_energy()
    ) > 0.5) - 1

    while (np.exp(-1 * atoms_prime.get_total_energy() +
                  atoms.get_total_energy())) ** a > 2 ** -a:
        print 'initial step size', a
        step_size *= 2 ** a
        atoms_prime = leapfrog(atoms, step_size)
    print step_size
    return step_size


def nuts(atoms, accept_target, iterations, p_scale=1, write_traj=None,
         escape_level=13, seed=None):
    """
    No U-Turn Sampling in the Canonical Ensemble, generating minima on the
    atoms' potential energy surface

    Parameters
    ----------
    atoms: ase.Atoms object
        The starting atomic configuration
    accept_target: float
        The target acceptance value, usually .65
    iterations: int
        Number of HMC iterations to perform
    p_scale: float, defaults to 1.
        Effective temperature, the random momentum scale
    write_traj: ase Trajectory object, optional
        The save trajectory to save the proposed configurations to
    seed: None or int
        The seed to use for the random number generation
    Returns
    -------
    list:
        List of atomic configurations, the trajectory
    """
    if seed is None:
        seed = np.random.randint(0, 2 ** 31)
    rs = RandomState(seed)
    if write_traj is not None:
        atoms.set_momenta(rs.normal(0, p_scale, (len(atoms), 3)))
        initial_vel = atoms.get_velocities()
        initial_forces = atoms.get_forces()
        initial_energy = atoms.get_potential_energy()
        write_traj.write(atoms)
    traj = [atoms]

    m = 0

    step_size = find_step_size(atoms, rs)
    mu = np.log(10 * step_size)
    # ebar = 1
    sim_hbar = 0
    gamma = 0.05
    t0 = 10
    # k = .75
    samples_total = 0
    print 'start hmc'
    try:
        while m <= iterations:
            print 'step', step_size / fs, 'fs'
            # sample r0
            atoms.set_momenta(rs.normal(0, p_scale, (len(atoms), 3)))

            # re-sample u, note we work in post exponential units:
            # [0, exp(-H(atoms0)] <= exp(-H(atoms1) >>> [0, 1] <= exp(-deltaH)
            u = rs.uniform(0, 1)

            # Note that because we need to calculate the difference between the
            # proposed energy and the current energy we declare it here,
            # preventing the need for multiple calls to the energy function
            e0 = traj[-1].get_total_energy()

            e = step_size
            n, s, j = 1, 1, 0
            neg_atoms = dc(atoms)
            pos_atoms = dc(atoms)
            while s == 1:
                v = rs.choice([-1, 1])
                if v == -1:
                    (neg_atoms, _, atoms_prime, n_prime, s_prime, a,
                     na) = buildtree(neg_atoms, u, v, j, e, e0, rs)
                else:
                    (_, pos_atoms, atoms_prime, n_prime, s_prime, a,
                     na) = buildtree(pos_atoms, u, v, j, e, e0, rs)

                if s_prime == 1 and rs.uniform() < min(1, n_prime * 1. / n):
                    traj += [atoms_prime]
                    if write_traj is not None:
                        atoms_prime.get_forces()
                        atoms_prime.get_potential_energy()
                        write_traj.write(atoms_prime)
                    atoms = atoms_prime
                n = n + n_prime
                span = pos_atoms.positions - neg_atoms.positions
                span = span.flatten()
                s = s_prime * (
                    span.dot(neg_atoms.get_velocities().flatten()) >= 0) * (
                        span.dot(pos_atoms.get_velocities().flatten()) >= 0)
                j += 1
                print 'iteration', m, 'depth', j, 'samples', 2 ** j
                samples_total += 2 ** j
                # Prevent run away sampling, EXPERIMENTAL
                # If we have generated 8192 samples,
                # then we are moving too slowly and should start a new iter
                # hopefully with a larger step size
                if j >= escape_level:
                    print 'jmax emergency escape at {}'.format(j)
                    s = 0
            w = 1. / (m + t0)
            sim_hbar = (1 - w) * sim_hbar + w * (accept_target - a / na)

            step_size = np.exp(mu - (m ** .5 / gamma) * sim_hbar)

            m += 1
    except KeyboardInterrupt:
        if m == 0:
            m = 1
        pass
    print '\n\n\n'
    print 'number of leapfrog samples', samples_total
    print 'number of successful leapfrog samples', len(traj) - 1
    print 'percent of good leapfrog samples', float(
        len(traj) - 1) / samples_total * 100, '%'
    print 'number of leapfrog per iteration, average', float(samples_total) / m
    print 'number of good leapfrog per iteration, average', float(
        len(traj) - 1) / m
    return traj, samples_total, float(samples_total) / m, seed


def buildtree(input_atoms, u, v, j, e, e0, rs):
    """
    Build the tree of samples for NUTS, recursively

    Parameters
    -----------
    input_atoms: ase.Atoms object
        The atoms for the tree
    u: float
        slice parameter, the baseline energy to compare against
    v: -1 or 1
        The direction of the tree leaves, negative means that we simulate
        backwards in time
    j: int
        The tree depth
    e: float
        The stepsize
    e0: float
        Current energy
    rs: numpy.random.RandomState object
        The random state object used to generate the random numbers.  Use of a
        unified random number generator with a known seed should help us to
        generate reproducible simulations
    Returns
    -------
    Many things
    """
    if j == 0:
        atoms_prime = leapfrog(input_atoms, v * e)
        neg_delta_energy = e0 - atoms_prime.get_total_energy()
        try:
            exp1 = np.exp(neg_delta_energy)
            exp2 = np.exp(Emax + neg_delta_energy)
        except:
            exp1 = 0
            exp2 = 0
        # print exp1, exp2
        # n_prime = int(u <= np.exp(-atoms_prime.get_total_energy()))
        # s_prime = int(u <= np.exp(Emax-atoms_prime.get_total_energy()))
        n_prime = int(u <= exp1)
        s_prime = int(u < exp2)
        return (atoms_prime, atoms_prime, atoms_prime, n_prime, s_prime,
                min(1, np.exp(-atoms_prime.get_total_energy() +
                              input_atoms.get_total_energy())), 1)
    else:
        (neg_atoms, pos_atoms, atoms_prime, n_prime, s_prime, a_prime,
         na_prime) = buildtree(input_atoms, u, v, j - 1, e, e0, rs)
        if s_prime == 1:
            if v == -1:
                (neg_atoms, _, atoms_prime_prime, n_prime_prime, s_prime_prime,
                 app, napp) = buildtree(neg_atoms, u, v, j - 1, e, e0, rs)
            else:
                (_, pos_atoms, atoms_prime_prime, n_prime_prime, s_prime_prime,
                 app, napp) = buildtree(pos_atoms, u, v, j - 1, e,
                                        # atoms0,
                                        e0, rs)

            if rs.uniform() < float(n_prime_prime / (
                    max(n_prime + n_prime_prime, 1))):
                atoms_prime = atoms_prime_prime

            a_prime = a_prime + app
            na_prime = na_prime + napp

            datoms = pos_atoms.positions - neg_atoms.positions
            span = datoms.flatten()
            s_prime = s_prime_prime * (
                span.dot(neg_atoms.get_velocities().flatten()) >= 0) * (
                span.dot(pos_atoms.get_velocities().flatten()) >= 0)
            n_prime = n_prime + n_prime_prime
        return (neg_atoms, pos_atoms, atoms_prime, n_prime, s_prime, a_prime,
                na_prime)
