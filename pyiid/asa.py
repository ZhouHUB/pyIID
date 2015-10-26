import math

from ase.data import *
import numpy as np
from asap3.analysis.particle import FullNeighborList
__doc__ = """

Calculate the accessible-surface area of atoms.

Uses the simple Shrake-Rupley algorithm, that generates a
relatively uniform density of dots over every atoms and
eliminates those within the sphere of another atom. The remaining
dots is used to calculate the area.

Reference: A. Shrake & J. A. Rupley. "Environment and Exposure to
Solvent of Protein Atoms. Lysozyme and Insulin." J Mol Biol. 79
(1973) 351- 371. """



def generate_sphere_points(n):
    """
  Returns list of coordinates on a sphere using the Golden-
  Section Spiral algorithm.
  """
    points = np.zeros((n, 3))
    inc = math.pi * (3 - math.sqrt(5))
    offset = 2 / float(n)
    for k in xrange(int(n)):
        y = k * offset - 1 + (offset / 2)
        r = math.sqrt(1 - y * y)
        phi = k * inc
        points[k, :] = [math.cos(phi) * r, y, math.sin(phi) * r]
    return points


def calculate_asa(atoms, probe, cutoff=None, tag=1, n_sphere_point=960):
    """
  Returns the accessible-surface areas of the atoms, by rolling a
  ball with probe radius over the atoms with their radius
  defined.
  """
    if cutoff is None:
        elements = list(set(atoms.numbers))
        cutoff = np.min(vdw_radii[elements]) * 2
    sphere_points = generate_sphere_points(n_sphere_point)

    const = 4.0 * math.pi / len(sphere_points)
    areas = []
    surface = []
    n_list = list(FullNeighborList(cutoff, atoms))
    for i, atom_i in enumerate(atoms):
        neighbor_indices = n_list[i]
        n_neighbor = len(neighbor_indices)
        j_closest_neighbor = 0
        radius = probe + vdw_radii[atom_i.number]

        n_accessible_point = 0
        for k in xrange(n_sphere_point):
            is_accessible = True
            test_point = sphere_points[k, :] / np.linalg.norm(
                sphere_points[k, :]) * radius + atom_i.position
            cycled_indices = range(j_closest_neighbor, n_neighbor)
            cycled_indices.extend(range(j_closest_neighbor))

            for j in cycled_indices:
                atom_j = atoms[int(neighbor_indices[j])]
                r = vdw_radii[atom_j.number] + probe
                diff = atom_j.position - test_point
                if np.dot(diff, diff) < r * r:
                    j_closest_neighbor = j
                    is_accessible = False
                    break
            if is_accessible:
                n_accessible_point += 1
                surface.append(test_point)

        area = const * n_accessible_point * radius * radius
        if area > 0:
            atoms[i].tag = tag
        areas.append(area)
    return areas, surface


if __name__ == '__main__':
    from ase.cluster.decahedron import Decahedron
    from ase.visualize import view
    from ase.visualize.mlab import plot
    import ase.io
    # atoms = Decahedron('Au', 2, 3, 1)
    atoms = ase.io.read(
        '/mnt/bulk-data/Dropbox/BNL_Project/xyz_files/Au102MBA44_Auonly.xyz')
    # atoms.rattle()
    calculate_asa(atoms, 1.4, 3.32)
    view(atoms[[atom.index for atom in atoms if atom.tag == 1]])
    view(atoms[[atom.index for atom in atoms if atom.tag == 0]])
