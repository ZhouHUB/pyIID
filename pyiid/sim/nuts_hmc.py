from copy import deepcopy as dc
from ase.units import fs
import numpy as np
from numpy.random import RandomState
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from pyiid.sim import leapfrog
from pyiid.sim import Ensemble

__author__ = 'christopher'
Emax = 200


def find_step_size(input_atoms, temp):
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
    MaxwellBoltzmannDistribution(atoms, temp=temp, force_temp=True)

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


def buildtree(input_atoms, u, v, j, e, e0, rs, stationary=False,
              zero_rotation=False):
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
        atoms_prime = leapfrog(input_atoms, v * e, stationary=stationary,
                               zero_rotation=zero_rotation)
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
         na_prime) = buildtree(input_atoms, u, v, j - 1, e, e0, rs,
                               stationary, zero_rotation)
        if s_prime == 1:
            if v == -1:
                (neg_atoms, _, atoms_prime_prime, n_prime_prime, s_prime_prime,
                 app, napp) = buildtree(neg_atoms, u, v, j - 1, e, e0, rs,
                                        stationary, zero_rotation)
            else:
                (_, pos_atoms, atoms_prime_prime, n_prime_prime, s_prime_prime,
                 app, napp) = buildtree(pos_atoms, u, v, j - 1, e, e0, rs,
                                        stationary, zero_rotation)

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


class NUTSCanonicalEnsemble(Ensemble):
    def __init__(self, atoms, restart=None, logfile=None, trajectory=None,
                 temperature=100,
                 stationary=False, zero_rotation=False, escape_level=13,
                 accept_target=.65, seed=None):
        Ensemble.__init__(self, atoms, restart, logfile, trajectory, seed)
        self.accept_target = accept_target
        self.step_size = find_step_size(atoms, temperature)
        self.mu = np.log(10 * self.step_size)
        # self.ebar = 1
        self.sim_hbar = 0
        self.gamma = 0.05
        self.t0 = 10
        # self.k = .75
        self.samples_total = 0
        self.temp = temperature
        self.stationary = stationary
        self.zero_rotation = zero_rotation
        self.escape_level = escape_level
        self.m = 0

    def step(self):
        print 'time step size', self.step_size / fs, 'fs'
        # sample r0
        MaxwellBoltzmannDistribution(self.traj[-1], self.temp, force_temp=True)
        # self.traj[-1].set_momenta(self.random_state.normal(0, 1, (
        #     len(self.traj[-1]), 3)))
        # re-sample u, note we work in post exponential units:
        # [0, exp(-H(atoms0)] <= exp(-H(atoms1) >>> [0, 1] <= exp(-deltaH)
        u = self.random_state.uniform(0, 1)

        # Note that because we need to calculate the difference between the
        # proposed energy and the current energy we declare it here,
        # preventing the need for multiple calls to the energy function
        e0 = self.traj[-1].get_total_energy()

        e = self.step_size
        n, s, j = 1, 1, 0
        neg_atoms = dc(self.traj[-1])
        pos_atoms = dc(self.traj[-1])
        while s == 1:
            v = self.random_state.choice([-1, 1])
            if v == -1:
                (neg_atoms, _, atoms_prime, n_prime, s_prime, a,
                 na) = buildtree(neg_atoms, u, v, j, e, e0,
                                 self.random_state, self.stationary,
                                 self.zero_rotation)
            else:
                (_, pos_atoms, atoms_prime, n_prime, s_prime, a,
                 na) = buildtree(pos_atoms, u, v, j, e, e0,
                                 self.random_state, self.stationary,
                                 self.zero_rotation)

            if s_prime == 1 and self.random_state.uniform() < min(
                    1, n_prime * 1. / n):
                self.traj += [atoms_prime]
                atoms_prime.get_forces()
                atoms_prime.get_potential_energy()
                self.call_observers()
            n = n + n_prime
            span = pos_atoms.positions - neg_atoms.positions
            span = span.flatten()
            s = s_prime * (
                span.dot(neg_atoms.get_velocities().flatten()) >= 0) * (
                    span.dot(pos_atoms.get_velocities().flatten()) >= 0)
            j += 1
            print 'depth', j, 'samples', 2 ** j
            self.samples_total += 2 ** j
            # Prevent run away sampling, EXPERIMENTAL
            # If we have generated s**self.escape_level samples,
            # then we are moving too slowly and should start a new iter
            # hopefully with a larger step size
            if j >= self.escape_level:
                print 'jmax emergency escape at {}'.format(j)
                s = 0
        w = 1. / (self.m + self.t0)
        self.sim_hbar = (1 - w) * self.sim_hbar + w * (self.accept_target - a /
                                                     na)

        self.step_size = np.exp(self.mu - (self.m ** .5 / self.gamma) *
                                self.sim_hbar)

        self.m += 1
