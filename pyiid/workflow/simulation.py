__author__ = 'christopher'
from ase.io.trajectory import PickleTrajectory

from pyiid.sim.nuts_hmc import nuts

from simdb.search import *
from simdb.insert import *
from simdb.handlers import FileLocation

from filestore.retrieve import handler_context
import filestore.commands as fsc


def run_simulation(sim):
    # Load info from simulation request
    sim_params, = find_simulation_parameter_document(_id=sim.params.id)

    # TODO: Throw in some statments about timeouts etc.
    iterations = sim_params.iterations
    target_acceptance = sim_params.target_acceptance
    ensemble_temp = sim_params.temperature

    # Load Atoms
    atoms_entry, = find_atomic_config_document(_id=sim.atoms.id)
    traj = atoms_entry.file_payload

    # now we have 3 options:
    # 1) we want to continue an existing simulation, which is a traj
    # 2) we want a new simulation, based on the first position of an
    #       existing traj
    # 3) we want a new simulation, based on an atomic configuration

    # if we are going to continue the simulation, and there is a simulation to
    # continue (more than one atomic configurations in the trajectory)

    # 1)
    if sim_params.continue_sim and isinstance(traj, list):
        # Give back the final configuration
        atoms = traj[-1]
        # Search filestore and get the file_location
        with handler_context({'ase': FileLocation}):
            atoms_file_location = fsc.retrieve(atoms_entry.file_uid)
        wtraj = PickleTrajectory(atoms_file_location, 'a')
    # 2)
    elif isinstance(traj, list) and not sim_params.continue_sim:
        # Give back the initial config
        atoms = traj[0]
        # Generate new file location and save it to filestore
        new_atoms_entry = insert_atom_document(
            atoms_entry.name + '_' + sim.name, atoms)

        with handler_context({'ase': FileLocation}):
            new_file_location = fsc.retrieve(new_atoms_entry.file_uid)
        wtraj = PickleTrajectory(new_file_location, 'w')
        sim.atoms = new_atoms_entry
        sim.save()
    # 3)
    else:
        atoms = traj[-1]
        # Search filestore and get the file_location
        with handler_context({'ase': FileLocation}):
            atoms_file_location = fsc.retrieve(atoms_entry.file_uid)
        wtraj = PickleTrajectory(atoms_file_location, 'a')

    # Create Calculators
    pes, = find_pes_document(_id=sim.pes.id)
    master_calc = pes.payload

    # Attach MulitCalc to atoms
    atoms.set_calculator(master_calc)

    sim.start_total_energy = atoms.get_total_energy()
    sim.start_potential_energy = atoms.get_potential_energy()
    sim.start_kinetic_energy = atoms.get_kinetic_energy()
    sim.start_time = ttime.time()
    sim.ran = True
    sim.save()

    # Simulate
    # TODO: eventually support different simulation engines
    out_traj, samples, l_p_i = nuts(atoms, target_acceptance, iterations,
                                    ensemble_temp, wtraj)
    sim.end_time = ttime.time()
    if 'total_iterations' in sim.__dict__.keys():
        if sim.total_iterations != 0:
            sim.total_iterations += sim.params.iterations
        else:
            sim.total_iterations = sim.params.iterations
    else:
        sim.total_iterations = sim.params.iterations
    if sim.total_samples is not None:
        sim.total_samples += samples
    else:
        sim.total_samples = samples
    sim.leapfrog_per_iter = l_p_i
    sim.finished = True
    sim.save()
    # Write info to DB
    sim.final_potential_energy = out_traj[-1].get_potential_energy()
    sim.final_kinetic_energy = out_traj[-1].get_kinetic_energy()
    sim.final_kinetic_energy = out_traj[-1].get_total_energy()
    sim.save()
