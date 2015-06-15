__author__ = 'christopher'
import os
from ase.io.trajectory import PickleTrajectory

from pyiid.wrappers.elasticscatter import ElasticScatter
from pyiid.sim.nuts_hmc import nuts

from pyiid.calc.multi_calc import MultiCalc

from pyiid.calc.pdfcalc import PDFCalc
from pyiid.calc.fqcalc import FQCalc
from pyiid.calc.spring_calc import Spring
from ase.calculators.lammpslib import LAMMPSlib

from simdb.odm_templates import Simulation
from simdb.search import *


def run_simulation(Simulation):

    # Load info from simulation request
    sim_params, = find_simulation_parameter_document(_id=Simulation.params.id)

    #TODO: Throw in some statments about timeouts etc.
    iterations = sim_params.iterations
    target_acceptance = sim_params.target_acceptance
    ensemble_temp = sim_params.temperature

    # Load Atoms
    atoms_entry, = find_atomic_config_document(_id=id(Simulation.atoms))
    traj = atoms_entry.file_payload

    # now we have 3 options:
    # 1) we want to continue an existing simulation, which is a traj
    # 2) we want a new simulation, based on the first position of an
    #       existing traj
    # 3) we want a new simulation, based on an atomic configuration

    # if we are going to continue the simulation, and there is a simulation to
    # continue (more than one atomic configurations in the trajectory)

    # 1)
    if sim_params.continue_sim and type(traj) == list:
        # Give back the final configuration
        atoms = traj[-1]
        # Search filestore and get the file_location

        wtraj = PickleTrajectory(atoms_file_location, 'a')
    # 2)
    elif type(traj) == list and not sim_params.continue_sim:
        # Give back the initial config
        atoms = traj[0]
        # Generate new file location and save it to filestore
        wtraj = PickleTrajectory(new_file_location, 'w')
    # 3)
    else:
        atoms = traj
        # Search filestore and get the file_location
        wtraj = PickleTrajectory(atoms_file_location, 'a')

    # Create Calculators
    pes, = find_pes_document(_id=Simulation.pes.id)
    master_calc = pes.payload

    # Attach MulitCalc to atoms
    atoms.append(master_calc)
    # Rattle atoms if built from scratch
    atoms.rattle()
    # Simulate
    # TODO: eventually support different simulation engines
    out_traj = nuts(atoms, target_acceptance, iterations,
                    ensemble_temp, wtraj)

# Write info to DB
