__author__ = 'christopher'

import numpy as np
from diffpy.Structure.structure import Structure
from diffpy.Structure.atom import Atom as dAtom
from ase.atoms import Atoms as AAtoms


def convert_atoms_to_stru(atoms):
    diffpy_atoms = []
    symbols = atoms.get_chemical_symbols()
    q = atoms.get_positions()
    tags = atoms.get_tags()
    for symbol, xyz, tag, in zip(symbols, q, tags):
        d_atom = dAtom(symbol, xyz=xyz,
                      label=tag, occupancy=1)
        diffpy_atoms.append(d_atom)
    stru = Structure(diffpy_atoms)
    return stru


def update_stru(new_atoms, stru):
    aatomq = new_atoms.get_positions()
    datomq = np.reshape([datom.xyz for datom in stru], (len(new_atoms), 3))
    # aatome = new_atoms.get_chemical_symbols()
    # datome = np.array([datom.element for datom in stru])
    changedq = np.in1d(aatomq, datomq).reshape((len(new_atoms), 3))

    changed_array = np.sum(changedq, 1) != 3
    stru[changed_array].xyz = new_atoms[changed_array].get_positions()
    # for i in len(changed_array):
    #     if changed_array[i] == True:
    #         stru[i]._set_xyz_cartn(new_atoms[i].position)
    # changed_list = []
    # for i in len(new_atoms):
    #     if np.sum(changedq[i, :]) != 3:
    #         changed_list.append(i)
    # for j in changed_list:
    #     stru[j]._set_xyz_cartn(new_atoms[j].position)
    return stru

def load_gr_file(gr_file=None, skiplines=None):
    #TODO: also give back the filename
    if gr_file is None:
        print 'Open Gr'
        gr_file = tkFileDialog.askopenfilename()
    if skiplines is None:
        with open(gr_file) as my_file:
            for num, line in enumerate(my_file,1):
                if '#### start data' in line:
                    skiplines=num+2
                    break
    data = np.loadtxt(gr_file, skiprows=skiplines)
    return data[:, 0], data[:, 1]