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

def run_simulation(Simulation):

    # Load info from simulation request
    params = Simulation.params

    #TODO: Throw in some statments about timeouts etc.
    iterations = params.iterations
    target_acceptance = params.targe_acceptance
    ensamble_temp = params.ensamble_temp

    # Load Atoms
    # Also create wtraj or pull this from the database?
    atoms_entry = Simulation.atoms
    if params.continue_sim:
        # Give back the final configuration
        atomic_config = atoms_entry.atomic_config
        wtraj = PickleTrajectory(atomic_config.filename, 'a')
    else:
        # Give back the initial config
        atomic_config = atoms_entry.atomic_config
        wtraj = PickleTrajectory(atomic_config.filename, 'w')

    # Create Calculators
    master_calc_list = []
    pes = Simulation.pes
    for calc in pes:
        # load the calculator with its kwargs
        # append it to the master_calc_list
        master_calc_list.append(calc)

        pass
    # Create MultiCalc
    master_calc = MultiCalc(calc_list=master_calc_list)

    # Attach MulitCalc to atoms
    atomic_config.append(master_calc)
    # Rattle atoms if built from scratch

    # Simulate
    out_traj = nuts(atomic_config, target_acceptance, iterations,
                    ensamble_temp, wtraj)

# Write info to DB
