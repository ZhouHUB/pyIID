__author__ = 'christopher'
import os
import time
import datetime
import math
from random import randint
import json
import pprint

import ase.io as aseio
from ase.atoms import Atoms
from ase.calculators.lammpslib import LAMMPSlib
from ase.io.trajectory import PickleTrajectory

from pyiid.wrappers.scatter import Scatter
from pyiid.calc.multi_calc import MultiCalc
from pyiid.calc.oo_pdfcalc import PDFCalc
from pyiid.calc.spring_calc import Spring
from pyiid.calc.oo_fqcalc import FQCalc
from pyiid.utils import load_gr_file



# We would like to have a way to setup and run the simulation, collecting
# metadata along the way, analyze the results, keep track of comments, and
# where all the files went/are going.  This should be loadable, giving us the
# ability to restart simulations, run the same simulation over again, and
# quickly grab the analysis figures/data.
# In short a proto-DB, possibly written in json.

def run_simulation(db_name, exp_type, exp_files, starting_structure, calcs,
                   sim_dict=None, exp_dict=None, rattle=(.001, 42),
                   comments=None):
    db_path = os.path.split(db_name)[0]
    run_db = {'exp_type': exp_type, 'exp_files': exp_files, 'exp_dict':exp_dict,
              'starting_structure': starting_structure, 'calcs': calcs,
              'sim_dict': sim_dict, 'comments': comments, 'rattle': rattle}
    try:
        # Load in the "experimental" data to match against
        fobs = None
        if run_db['exp_type'] == 'theory':
            # Load NP structure
            assert type(run_db['exp_files']) == str
            th_atoms = aseio.read(run_db['exp_files'])

            # Get Gobs, Fobs
            s = Scatter(run_db['exp_dict'])
            gobs = s.get_pdf(th_atoms)
            fobs = s.get_fq(th_atoms)

        elif run_db['exp_type'] == 'x-ray total scatter':
            # TODO: load the data depending on extension
            r, gobs, run_db['exp_dict'] = load_gr_file(run_db['exp_files'])
            # TODO: F(Q) loading not supported yet!
            # fobs, exp_dict_fq =
            s = Scatter(run_db['exp_dict'])

        else:
            raise NotImplementedError

        run_db['exp_dict'] = s.exp

        # Build starting candidate structure
        if type(starting_structure) is str:
            # Load the file as atoms
            if os.path.splitext(starting_structure)[-1] == '.traj':
                wtraj = PickleTrajectory(starting_structure, 'a')
            else:
                while True:
                    ID_number = randint(0, 10000)

                    wtraj_name = \
                        os.path.splitext(
                            os.path.split(starting_structure)[-1])[0]
                    wtraj_name = os.path.join(db_path, wtraj_name)
                    wtraj_name += '_' + str(ID_number) + '.traj'
                    if os.path.exists(wtraj_name) is False:
                        break
                wtraj = PickleTrajectory(wtraj_name, 'w')

            start_atoms = aseio.read(starting_structure)
        elif type(starting_structure) is Atoms:
            start_atoms = starting_structure

            prefix = ''
            for calc in calcs:
                prefix += calc['name'] + '_'
            while True:
                ID_number = randint(0, 10000)
                wtraj_name = os.path.join(db_path, prefix)
                wtraj_name += str(ID_number) + '.traj'
                if os.path.exists(wtraj_name) is False:
                    break
            wtraj = PickleTrajectory(wtraj_name, 'w')
        else:
            raise NotImplementedError

        run_db['ID number'] = ID_number
        run_db['run path'] = os.getcwd()
        run_db['traj loc'] = wtraj_name

        if run_db['rattle'] is not None and run_db['rattle'] is not False:
            start_atoms.rattle(*rattle)

        # Get calculators ready
        supported_calcs = {'PDF': PDFCalc, 'FQ': FQCalc, 'Spring': Spring,
                           'LAMMPS': LAMMPSlib}
        calc_l = []
        for calc_dict in calcs:
            if calc_dict['name'] in supported_calcs.keys():
                if calc_dict['name'] is 'PDF':
                    calc_dict['kwargs']['gobs'] = gobs
                    calc_dict['kwargs']['scatter'] = s
                if calc_dict['name'] is 'FQ' and fobs is not None:
                    calc_dict['kwargs']['scatter'] = s
                    calc_dict['kwargs']['fobs'] = fobs

                calc_l.append(
                    supported_calcs[calc_dict['name']](**calc_dict['kwargs']))
            else:
                raise NotImplementedError

        calc = MultiCalc(calc_list=calc_l)
        start_atoms.set_calculator(calc)
        print 'Total energy', start_atoms.get_total_energy()

        run_db['Start Total Energy'] = start_atoms.get_total_energy()
        run_db['Start Potential Energy'] = start_atoms.get_potential_energy()
        run_db['Start Kinetic Energy'] = start_atoms.get_kinetic_energy()

        if run_db['sim_dict'] is not None:
            # Prep the Simulation
            if run_db['sim_dict']['Simulation type'] == 'NUTS-HMC':
                from pyiid.sim.nuts_hmc import nuts
                # Prep for NUTS-HMC
                pe_list = []
                ti = time.time()
                traj = nuts(start_atoms, *run_db['sim_dict']['Sim args'],
                            wtraj=wtraj)
                tf = time.time()

                run_db['Time to completion'] = str(
                    datetime.timedelta(seconds=math.ceil(tf - ti)))
                run_db['Final Total Energy'] = traj[-1].get_total_energy()
                run_db[
                    'Final Potential Energy'] = traj[-1].get_potential_energy()
                run_db[
                    'Final Kinetic Energy'] = traj[-1].get_kinetic_energy()
    except KeyboardInterrupt:
        pass
    # clean up NP arrays
    for calc_dict in calcs:
        if calc_dict['name'] in supported_calcs.keys():
            if calc_dict['name'] is 'PDF':
                del calc_dict['kwargs']['gobs']
                del calc_dict['kwargs']['scatter']
            if calc_dict['name'] is 'FQ' and fobs is not None:
                del calc_dict['kwargs']['scatter']
                del calc_dict['kwargs']['fobs']

    pprint.pprint(run_db)
    with open(db_name, 'a') as f:
        f.write(json.dumps(run_db))
        f.write('\n')


def restart_sim(db_name, db_entry):
    run_db = db_entry
    try:
        # Load in the "experimental" data to match against
        fobs = None
        if run_db['exp_type'] == 'theory':
            # Load NP structure
            assert type(run_db['exp_files']) == str
            th_atoms = aseio.read(run_db['exp_files'])

            # Get Gobs, Fobs
            s = Scatter(run_db['exp_dict'])
            gobs = s.get_pdf(th_atoms)
            fobs = s.get_fq(th_atoms)

        elif run_db['exp_type'] == 'x-ray total scatter':
            # TODO: load the data depending on extension
            # Load the data and chop it?
            gobs, run_db['exp_dict'] = load_gr_file(run_db['exp_files'])
            # F(Q) loading not supported yet!
            # fobs, exp_dict_fq =
            s = Scatter(run_db['exp_dict'])

        else:
            raise NotImplementedError
        run_db['exp_dict'] = s.exp

        # Build starting candidate structure
        start_atoms = aseio.read(run_db['traj loc'])
        wtraj = PickleTrajectory(db_entry['traj loc'], 'a')

        # Get calculators ready
        supported_calcs = {'PDF': PDFCalc, 'FQ': FQCalc, 'Spring': Spring,
                           'LAMMPS': LAMMPSlib}
        calc_l = []
        for calc_dict in calcs:
            if calc_dict['name'] in supported_calcs.keys():
                if calc_dict['name'] is 'PDF':
                    calc_dict['kwargs']['gobs'] = gobs
                    calc_dict['kwargs']['scatter'] = s
                if calc_dict['name'] is 'FQ' and fobs is not None:
                    calc_dict['kwargs']['scatter'] = s
                    calc_dict['kwargs']['fobs'] = fobs

                calc_l.append(
                    supported_calcs[calc_dict['name']](**calc_dict['kwargs']))
            else:
                raise NotImplementedError
        calc = MultiCalc(calc_list=calc_l)
        start_atoms.set_calculator(calc)
        print 'Total energy', start_atoms.get_total_energy()

        if run_db['sim_dict'] is not None:
            # Prep the Simulation
            if run_db['sim_dict']['Simulation type'] == 'NUTS-HMC':
                from pyiid.sim.nuts_hmc import nuts
                # Prep for NUTS-HMC
                pe_list = []
                ti = time.time()
                traj = nuts(start_atoms, *run_db['sim_dict']['Sim args'],
                            wtraj=wtraj)
                tf = time.time()

                run_db['Time to completion'] = str(
                    datetime.timedelta(seconds=math.ceil(tf - ti)))
                run_db['Final Total Energy'] = traj[-1].get_total_energy()
                run_db[
                    'Final Potential Energy'] = traj[-1].get_potential_energy()
                run_db[
                    'Final Kinetic Energy'] = traj[-1].get_kinetic_energy()
    except KeyboardInterrupt:
        pass

    for calc_dict in calcs:
        if calc_dict['name'] in supported_calcs.keys():
            if calc_dict['name'] is 'PDF':
                del calc_dict['kwargs']['gobs']
                del calc_dict['kwargs']['scatter']
            if calc_dict['name'] is 'FQ' and fobs is not None:
                del calc_dict['kwargs']['scatter']
                del calc_dict['kwargs']['fobs']

    pprint.pprint(run_db)
    with open(db_name, 'a') as f:
        f.write(json.dumps(run_db))
        f.write('\n')

if __name__ == '__main__':
    from pyiid.utils import build_sphere_np
    exp_dict = {'qmin': 0.0, 'qmax': 25., 'qbin': .1, 'rmin': 2.45,
                'rmax': 25., 'rstep': .01}
    calcs = [
        {'name': 'PDF', 'kwargs': {'conv': 300, 'potential': 'rw'}},
        # {'name': 'FQ', 'kwargs': {'conv': 50, 'potential': 'rw'}},
        {'name': 'Spring', 'kwargs': {'k': 100, 'rt': exp_dict['rmin']}}
    ]
    '''
    run_simulation(
        '/mnt/work-data/dev/IID_data/db_test/test.json',
        'theory',

        # '/mnt/work-data/dev/IID_data/examples/Au/736_atom/DFT_crystal_v_disorder/Au736_disordered.xyz',
        # '/mnt/work-data/dev/IID_data/examples/Au/736_atom/DFT_crystal_v_disorder/Au736_crystalline.xyz',

        '/mnt/work-data/dev/IID_data/examples/Au/55_amorphous/Au55.300K_amorphous.xyz',
        '/mnt/work-data/dev/IID_data/examples/Au/55_amorphous/Au55.xyz',

        # '/mnt/work-data/dev/IID_data/examples/C/buckyball/C60.xyz',
        # '/mnt/work-data/dev/IID_data/examples/C/buckyball/C60.xyz',

        calcs,
        {'Simulation type': 'NUTS-HMC', 'Sim args': (.65, 100, 1)},
        exp_dict,
        rattle=(.001, 42),
        comments='Au55 with Spring, Hot'
    )
    '''

    # 2nm Au
    atomsio = build_sphere_np('/mnt/work-data/dev/IID_data/examples/Au/2_nm/1100138.cif', 20/2.)

    run_simulation(
        '/mnt/work-data/dev/IID_data/db_test/test.json',
        'x-ray total scatter',
        '/mnt/work-data/dev/IID_data/examples/Au/2_nm/10_112_15_Au_Fit2d_FinalSum.gr',
        atomsio,
        calcs,
        {'Simulation type': 'NUTS-HMC', 'Sim args': (.65, 100, 1.)},
        exp_dict,
        rattle=(.1, 0),
        comments='2nm Au with Spring'
    )
        # '''