__author__ = 'christopher'
import numpy as np
from ase.atoms import Atoms as atoms
from copy import deepcopy as dc


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
    f = atoms.get_forces()
    atoms.set_velocities(atoms.get_velocities() + step * f)
    new_pos = atoms.get_positions() + step * atoms.get_velocities()
    atoms.set_positions(new_pos)
    return atoms


def simulate_dynamics(atoms, stepsize, n_steps):
    """
    Create a new atomic configuration by simulating the hamiltonian dynamics
    of the system

    Parameters
    ----------
    atoms: ase.Atoms ase.Atoms
        The atomic configuration
    stepsize: float
        The step size for the simulation
    n_steps: int
        The number of steps

    Returns
    -------
     list of ase.Atoms
        This list contains all the moves made during the simulation
    """
    # get the initial positions and momenta
    grad = atoms.get_forces()
    moves = [dc(atoms)]
    # moves = None
    # print moves[0].get_positions()
    atoms.set_velocities(atoms.get_velocities() + 0.5 * stepsize * grad)
    atoms.set_positions(
        atoms.get_positions() + stepsize * atoms.get_velocities())
    # print atoms.get_positions()
    moves += [dc(atoms)]
    for n in range(n_steps):
        f1 = atoms.get_forces()
        prop_move = [dc(leapfrog(atoms, stepsize))]
        f2 = atoms.get_forces()
        '''
        if np.sum(f2**2) > np.sum(f1**2):
            print 'stop short at n=', n
            atoms = moves[-1]
            break
        else:
            moves += prop_move
        '''
    atoms.set_velocities(atoms.get_velocities() + 0.5 * stepsize *
                         atoms.get_forces())
    return moves


def mh_accept(inital_energy, next_energy):
    """
    Decide wheter to accept or reject the new configuration

    Parameters
    -----------
    initial_energy: float
        The Hamiltonian of the intial configuration
    next_energy: float
        The Hamiltonian of the proposed configuration

    Returns
    -------
    np._bool
        Whether to accept the new configuration
    """
    diff = inital_energy - next_energy
    ave = np.average([inital_energy, next_energy])
    rand = np.random.random((1,))
    # expe = np.exp(diff / ave)
    expe = np.exp(diff)
    print 'initial, next, rand, exp'
    print inital_energy, next_energy, rand, expe
    return rand < expe
    # print np.exp(diff) - np.random.random(inital_energy.shape)
    # return np.exp(diff) - np.random.random(inital_energy.shape) >= 0


def kin_energy(atoms):
    """
    Calculate the kinetic energy of the atoms

    Parameters
    -----------
    atoms: ase.Atoms ase.Atoms
        The atomic configuration
    Returns
    -------
    float
        The kinetic energy
    """
    v = atoms.get_velocities()
    if v is None:
        v = 0.0
    return np.sum(v ** 2) / 2.0


def hamil(atoms):
    """
    Calculate the Hamiltonian for the atomic configuration 

    Parameters
    ----------
    atoms: ase.Atoms
        The atomic configuration

    Return
    ------
    float
        The energy of the system
    
    """
    print 'pot', atoms.get_potential_energy()
    print 'kin', kin_energy(atoms)
    return atoms.get_potential_energy() \
           + kin_energy(atoms)


def hmc_move(atoms, stepsize, n_steps):
    """
    Move atoms and check if they are accepted, returning the new 
    configuration if accepted, the old if not
    
    Parameters
    -----------
    atoms: ase.Atoms 
    stepsize: 
    n_steps:
    
    Returns
    --------
    bool
        True if new atoms accepted, false otherwise
    ase.atoms
        The atomic configuration to be used in the next iteration
    
    """
    atoms.set_velocities(np.random.normal(0, 1, (len(atoms), 3)))
    prop = dc(atoms)
    moves = simulate_dynamics(prop, stepsize, n_steps)
    accept = mh_accept(hamil(atoms), hamil(prop))
    # print accept
    # print(accept is 0)
    # assert(type(accept) == bool)
    if accept == 1:
        # print 'True'
        return True, prop
    elif accept == 0:
        # print('false')
        return False, atoms


def run_hmc(atoms, iterations, stepsize, n_steps, avg_acceptance_slowness,
            avg_acceptance_rate, target_acceptance_rate, stepsize_inc,
            stepsize_dec, stepsize_min, stepsize_max):
    """
    Wrapper for running Hamiltonian (Hybrid) Monte Carlo refinements, using a dynamic stepsize refinement, based on whether moves are being accepted
    
    Parameters
    ----------
    atoms: ase.Atoms
    iterations: int
        The number of
    stepsize: float
        The stepsize for the simulation
    n_steps: int
        The number of steps to take during the simulation
    avg_acceptance_slowness: float
        The desired time constant
    avg_acceptance_rate: float
    target_acceptance_rate: float
        The desired acceptance rate
    stepsize_inc: float
        The amount by which the stepsize is increased
    stepsize_dec: float
        The amount by which the stepsize is decreased
    stepsize_min: float
        The minimum stepsize
    stepsize_max: float
        The maximum stepsize

    Returns
    -------
    traj: list of ase.Atoms
        The list of configurations
    accept_list: list of bools
        The list describing whether moves were accpeted or rejected
    move_list: ??? (soon to be eliminated)

    """
    traj = [atoms]
    accept_list = []
    move_list = []
    # initial_energy = atoms.get_potential_energy()
    i = 0
    try:
        while i < iterations:
            accept, atoms = hmc_move(atoms, stepsize, n_steps)
            print i, accept, atoms.get_potential_energy() * 10000, stepsize
            print '--------------------------------'
            if accept is True:
                traj += [atoms]
            accept_list.append(accept)
            avg_acceptance_rate = avg_acceptance_slowness * avg_acceptance_rate + (

                                                                                      1.0 - avg_acceptance_slowness) * np.average(
                np.array(accept_list))
            if avg_acceptance_rate > target_acceptance_rate:
                stepsize *= stepsize_inc
            else:
                stepsize *= stepsize_dec
            if stepsize > stepsize_max:
                stepsize = stepsize_max
            elif stepsize < stepsize_min:
                stepsize = stepsize_min

            # print i
            i += 1
    except KeyboardInterrupt:
        pass
    return traj, accept_list, move_list


if __name__ == '__main__':
    from ase.io.trajectory import PickleTrajectory
    from pyiid.wrappers.kernel_wrap import wrap_rw, wrap_pdf
    from pyiid.pdfcalc import PDFCalc
    from copy import deepcopy as dc
    import ase.io as aseio
    from pyiid.utils import load_gr_file
    import matplotlib.pyplot as plt


    atomsi = aseio.read('/home/christopher/pdfgui_np_25_rattle1_cut.xyz')
    # atomsi.rattle(.1)
    # atomsi = Atoms('Au4', [(0,0,0), (3,0,0), (0,3,0), (3,3,0)])
    # atomsi = Atoms('Au2', [(0,0,0), (3,0,0)])
    # pdf, fq = wrap_pdf(atomsi,
    # Qmin=2.5
    # )
    atomsi.rattle(.05)
    r, pdf = load_gr_file('/home/christopher/7_7_7_FinalSum.gr')
    pdf = pdf[:-1]
    rw, scale, apdf, afq = wrap_rw(atomsi, pdf, qmin=2.5)
    # plt.plot(apdf*scale, label='initial')
    calc = PDFCalc(gobs=pdf, qmin=2.5, conv=.0001, qbin=.1)
    atomsi.set_calculator(calc)
    # print(atomsi.get_potential_energy()*10000)
    # plt.plot(pdf, label='experiment')
    # plt.legend()
    # plt.show()
    # AAA
    # atoms2 = dc(atomsi)
    # atoms2[1].position = (3.05,0,0)
    # atoms2[2].position = (0,3.,0)
    # pdf2, fq2 = wrap_pdf(atoms2,
    # Qmin=2.5
    # )
    rwi = atomsi.get_potential_energy()
    print(rwi * 10000)
    atomsi.set_velocities(np.zeros((len(atomsi), 3)))

    # traj = simulate_dynamics(atoms2, .005, 20)
    pe_list = []


    # '''
    # t0 = time()
    # atomsi.get_forces()
    # print(time()-t0)
    # AAA
    traj, accept_list, move_list = run_hmc(atomsi, 100, .005, 5, 0.9, 0, .9,
                                           1.02, .98, .001, .65)

    for atoms in traj:
        pe_list.append(atoms.get_potential_energy())
    # print(accept_list)
    # print(traj)
    print((rwi - traj[-1].get_potential_energy()) * 10000)

    wtraj = PickleTrajectory(
        '/home/christopher/dev/pyIID/extra/nipd_test'
        '.traj'
        , 'w')
    for atoms in traj:
        wtraj.write(atoms)
        # print atoms.get_positions()
    # '''
    # pdf, fq = wrap_pdf(atomsi,
    # Qmax = 100
    # )
    alp = 500
    # '''
    i = 0

    plt.plot(pe_list)
    plt.show()
    '''
    for atoms in traj:
        rw, scale, pdf2, fq2 = wrap_rw(atoms=atoms, gobs=pdf)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(str(i)+','+str(scale))
        a, = ax.plot(pdf[:alp], label = 'initial')
        b, = ax.plot(pdf2[:alp]*scale, label = 'final')
        c, = ax.plot(pdf[:alp]-pdf2[:alp]*scale, label = 'diff')
        plt.legend(handles=[a, b, c])
        plt.show()
        i += 1

'''
    # '''
    pdf3, fq3 = wrap_pdf(traj[-1], )
    a, = plt.plot(pdf, label='initial')
    b, = plt.plot(pdf2, label='pre-hmc')
    d, = plt.plot(pdf3, label='post-hmc')
    c, = plt.plot(pdf - pdf3, label='initial-post')
    plt.legend(handles=[a, b, d, c])
    plt.show()
    '''
    a, = plt.plot(fq, label = 'initial')
    b, = plt.plot(fq2, label = 'final')
    d, = plt.plot(fq3, label = 'post-hmc')
    c, = plt.plot(fq-fq3, label = 'initial-post')
    plt.legend(handles=[a, b, d, c])
    plt.show()
    '''
    plt.plot(accept_list)
    plt.show()