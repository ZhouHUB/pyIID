__author__ = 'christopher'

import numpy as np
from diffpy.Structure.structure import Structure
from diffpy.Structure.atom import Atom as dAtom
from ase.atoms import Atoms as AAtoms
import ase.io as aseio
from ase.lattice.surface import add_adsorbate

import math
import tkFileDialog
from pyiid.kernels.serial_kernel import get_d_array
from copy import deepcopy as dc

def convert_atoms_to_stru(atoms):
    """
    Convert between ASE and Diffpy structural objects

    Parameters:
    -----------
    atoms: ase.Atoms object

    Return:
    diffpy.Structure object:
    """
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
    """
    Load gr files produced from PDFgetx3
    """
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


def convert_stru_to_atoms(stru):
    symbols = []
    xyz = []
    tags = []
    for d_atom in stru:
        symbols.append(d_atom.element)
        xyz.append(d_atom.xyz)
        tags.append(d_atom.label)
    # print symbols
    # print np.array(xyz)
    # print tags
    atoms = AAtoms(symbols, np.array(xyz), tags=tags)
    return atoms


def build_sphere_np(file, radius):
    atoms = aseio.read(file)
    cell_dist = atoms.get_cell()
    multiple = np.ceil(2 * radius / cell_dist)
    atoms.repeat((multiple[0,0], multiple[1,1], multiple[2,2]))
    com = atoms.get_center_of_mass()
    atoms.translate(-com)
    del atoms[[atoms.index for atom in atoms
               if np.sqrt(np.dot(atom.position, atom.position)) >=
               np.sqrt(radius**2)]]
    return atoms


def tag_surface_atoms(atoms, tag=1, tol=0):
    """
    Find which are the surface atoms in a nanoparticle.
    ..note: We define the surface as a collection of atoms for which the
    dot product between the displacement from the center of mass and all
    the interatomic distances are negative.  Thus, all the interatomic
    displacement vectors point inward or perpendicular.  This only works for
    true crystals so we will include some tolerance to account for surface
    disorder.
    :param atoms:
    :return:
    """
    n = len(atoms)
    q = atoms.get_positions()
    d_array = np.zeros((n, n, 3))
    get_d_array(d_array, q)
    com = atoms.get_center_of_mass()
    for i in range(n):
        pos = atoms[i].position
        disp = pos - com
        disp /= np.linalg.norm(disp)
        dot = np.zeros((n))
        for j in range(n):
            if j != i:
                dot[j] = np.dot(disp, d_array[i, j, :]/np.linalg.norm(d_array[i, j, :]))
        if np.all(np.less_equal(dot, np.zeros(len(dot))+tol)):
            atoms[i].tag = tag


def add_ligands(ligand, surface, distance, coverage, head, tail):
    atoms = dc(surface)
    tag_surface_atoms(atoms)
    for atom in atoms:
        if atom.tag == 1 and np.random.random() < coverage:
            pos = atom.position
            com = surface.get_center_of_mass()
            disp = pos-com
            norm_disp = disp/np.sqrt(np.dot(disp, disp))
            l_length = ligand[tail].position - ligand[head].position
            norm_l_length = l_length/np.sqrt(np.dot(l_length, l_length))
            ads = dc(ligand)
            ads.rotate(norm_l_length, a=norm_disp)
            ads.translate(-ads[head].position)
            ads.translate(pos + distance * norm_disp)
            atoms += ads
    return atoms