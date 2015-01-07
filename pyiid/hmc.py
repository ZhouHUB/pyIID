__author__ = 'christopher'
import numpy as np
from ase.atoms import Atoms
from copy import deepcopy as dc
atoms = Atoms


def leapfrog(atoms, step):
    F = atoms.get_forces()
    atoms.set_velocities(atoms.get_velocities() + step*F)
    new_pos = atoms.get_positions() + step*atoms.get_velocities()
    atoms.set_positions(new_pos)
    return atoms


def simulate_dynamics(atoms, stepsize, n_steps):
    #get the initial positions and momenta
    grad = atoms.get_forces()
    moves = [dc(atoms)]
    # moves = None
    # print moves[0].get_positions()
    atoms.set_velocities(atoms.get_velocities() + 0.5 * stepsize * grad)
    atoms.set_positions(atoms.get_positions() + stepsize * atoms.get_velocities())
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
    diff = inital_energy - next_energy
    ave = np.average([inital_energy, next_energy])
    rand = np.random.random((1,))
    expe = np.exp(diff/ave)
    print 'initial, next, rand, exp'
    print inital_energy, next_energy, rand, expe
    return rand < expe
    # print np.exp(diff) - np.random.random(inital_energy.shape)
    # return np.exp(diff) - np.random.random(inital_energy.shape) >= 0


def kin_energy(atoms):
    v = atoms.get_velocities()
    if v is None:
        v = 0.0
    return np.sum(v ** 2) / 2.0

def hamil(atoms):
    return atoms.get_potential_energy() \
           + kin_energy(atoms)


def hmc_move(atoms, stepsize, n_steps):
    atoms.set_velocities(np.random.normal(0, .1, (len(atoms), 3)))
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
    traj = [atoms]
    accept_list = []
    move_list = []
    # initial_energy = atoms.get_potential_energy()
    i = 0
    try:
        while i < iterations:
            accept, atoms = hmc_move(atoms, stepsize, n_steps)
            print i, accept, atoms.get_potential_energy()*10000, stepsize
            print '--------------------------------'
            if accept is True:
                traj += [atoms]
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
    except KeyboardInterrupt:
        pass
    return traj, accept_list, move_list

if __name__ == '__main__':
    from ase import Atoms
    from ase.io.trajectory import PickleTrajectory
    from pyiid.kernel_wrap import wrap_rw, wrap_grad_rw, wrap_pdf
    from pyiid.pdf_calc import pdf_calc
    from copy import deepcopy as dc
    import ase.io as aseio
    # atomsi = aseio.read('/home/christopher/pdfgui_np_25_rattle1_cut.xyz')
    # atomsi = Atoms('Au4', [(0,0,0), (3,0,0), (0,3.5,0), (3,3,0)])
    atomsi = Atoms('Au2', [(0,0,0), (3,0,0)])
    pdf, fq = wrap_pdf(atomsi,
                       # Qmin=2.5
    )
    calc = pdf_calc(gobs=pdf,
                    conv=.0001,
                    # Qmax = 100
                    # Qmin=2.5
    )
    atomsi.set_calculator(calc)
    atoms2 = dc(atomsi)
    atoms2[1].position = (3.08,0,0)
    # atoms2[2].position = (0,3.,0)
    pdf2, fq2 = wrap_pdf(atoms2,
                       # Qmin=2.5
    )
    rwi = atoms2.get_potential_energy()
    print(rwi*10000)
    atoms2.set_velocities(np.zeros((len(atoms2), 3)))

    # traj = simulate_dynamics(atoms2, .005, 20)
    pe_list = []


    # '''
    traj, accept_list, move_list = run_hmc(atoms2, 50, .005, 5, 0.9, 0, .9,
                                           1.02, .98, .001, .25)

    for atoms in traj:
        pe_list.append(atoms.get_potential_energy())
    # print(accept_list)
    # print(traj)
    print(rwi - traj[-1].get_potential_energy())

    wtraj = PickleTrajectory(
        '/home/christopher/dev/pyIID/extra/au_two'
                             '.traj'
                             ,'w')
    for atoms in traj:
        wtraj.write(atoms)
        # print atoms.get_positions()
    # '''
    # pdf, fq = wrap_pdf(atomsi,
    #                    Qmax = 100
    # )
    alp = 500
    # '''
    i = 0
    import matplotlib.pyplot as plt
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
    pdf3, fq3 = wrap_pdf(traj[-1],)
    a, = plt.plot(pdf, label = 'initial')
    b, = plt.plot(pdf2, label = 'pre-hmc')
    d, = plt.plot(pdf3, label = 'post-hmc')
    c, = plt.plot(pdf-pdf3, label = 'initial-post')
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