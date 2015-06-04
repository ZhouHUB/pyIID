__author__ = 'christopher'
import json
import os
from ase.io.trajectory import PickleTrajectory
import ase.io as aseio

from pyiid.wrappers.scatter import Scatter


from pyiid.calc.multi_calc import MultiCalc

from pyiid.calc.oo_pdfcalc import PDFCalc
from pyiid.calc.oo_fqcalc import FQCalc
from pyiid.calc.spring_calc import Spring
from ase.calculators.lammpslib import LAMMPSlib

supported_calcs = {'PDF': PDFCalc, 'FQ': FQCalc, 'Spring': Spring,
                           'LAMMPS': LAMMPSlib}
pcalc = PDFCalc()



load_pdf_dict = {
    'generation':'theory',
    'file location': str,
    'rmin': None,
    'rmax': None
}

spring_d = {
    'kwargs': {'k': 100}
}

pdf_calc_dict = {'load':load_pdf_dict, 'kwargs': {'conv': 300, 'potential': 'rw'}}

request = {
    'PES':[
        {'PES name': 'PDF', 'Calc dict': pdf_calc_dict},
        {'PES name': 'Spring', 'Calc dict': spring_d}
    ],
    'start config':None,
    'sim_dict':None,
    'comments':None
}


# Load info from simulation request
def load_db_entry(db_str):
    db_entry = json.load(db_str)

    atoms_file =  db_entry['start config']
    atoms = aseio.read(atoms_file)
    ext = os.path.splitext(atoms_file)[-1]
    if ext == '.traj':
        wtraj = PickleTrajectory(atoms_file, 'a')
    else:
        wtraj = None
    d = {'atoms': atoms, 'wtraj': wtraj}


# Create Scatter object

# Create calculators
calcs = []
for PES_dict in request['PES']:
    if not PES_dict['name'] in supported_calcs.keys():
        raise NotImplementedError

    if PES_dict['name'] is 'PDF':
        calc_dict = PES_dict['Calc dict']

        # Load/create G(r)
        r, gobs, exp_dict = get_gr(**calc_dict['load'])
        s = Scatter(exp_dict)
        calc = PDFCalc(gobs=, scatter=, **calc['PES dict'])
        calc_dict['kwargs']['gobs'] = gobs
        calc_dict['kwargs']['scatter'] = s

    if calc_dict['name'] is 'FQ' and fobs is not None:
        # Load/create F(Q)

        calc_dict['kwargs']['scatter'] = s
        calc_dict['kwargs']['fobs'] = fobs

    if calc_dict['name'] is 'Spring':
        calc = Spring(rt=s.exp['rmin'], **calc_dict['PES dict'])

    calcs.append(calc)
# Create MultiCalc
calc = MultiCalc(calc_list=calcs)

# Load Atoms
# Rattle atoms if needed
# Attach MulitCalc to atoms

# Create wtraj
# Get Sim parameters

# Simulate

# Write info to DB

