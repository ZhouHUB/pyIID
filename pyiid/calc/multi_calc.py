__author__ = 'christopher'
from ase.calculators.calculator import Calculator
import numpy as np


class MultiCalc(Calculator):
    """
    Class for doing multiple calculator energy calculations
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, restart=None, ignore_bad_restart_file=False, label=None,
                 atoms=None, calc_list=None, **kwargs):

        Calculator.__init__(self, restart, ignore_bad_restart_file,
                            label, atoms, **kwargs)

        self.calc_list = calc_list

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=['positions', 'numbers', 'cell',
                                  'pbc', 'charges', 'magmoms']):
        """PDF Calculator

        atoms: Atoms object
            Contains positions, unit-cell, ...
        properties: list of str
            List of what needs to be calculated.  Can be any combination
            of 'energy', 'forces'
        system_changes: list of str
            List of what has changed since last calculation.  Can be
            any combination of these five: 'positions', 'numbers', 'cell',
            'pbc', 'charges' and 'magmoms'.
            """

        Calculator.calculate(self, atoms, properties, system_changes)

        # we shouldn't really recalc if charges or magmos change
        if len(system_changes) > 0:  # something wrong with this way
            if 'energy' in properties:
                self.calculate_energy(self.atoms)

            if 'forces' in properties:
                self.calculate_forces(self.atoms)
        for property in properties:
            if property not in self.results:
                if property is 'energy':
                    self.calculate_energy(self.atoms)

                if property is 'forces':
                    self.calculate_forces(self.atoms)

    def calculate_energy(self, atoms):
        """
        Calculate energy
        :param atoms:
        :return:
        """
        energy_list = []
        for calc in self.calc_list:
            atoms.set_calculator(calc)
            energy_list.append(atoms.get_potential_energy())
            atoms._del_calculator
        energy = sum(energy_list)
        self.energy_free = energy
        self.energy_zero = energy
        self.results['energy'] = energy

    def calculate_forces(self, atoms):
        self.results['forces'] = np.zeros((len(atoms), 3))
        forces = np.zeros((len(atoms), 3))
        for calc in self.calc_list:
            atoms.set_calculator(calc)
            forces[:,:] += atoms.get_forces()
            atoms._del_calculator
        self.results['forces'] = forces


if __name__ == '__main__':
    from ase.atoms import Atoms
    from ase.visualize import view
    from pyiid.calc.pdfcalc import PDFCalc
    from pyiid.wrappers.multi_gpu_wrap import wrap_pdf
    from pyiid.wrappers.kernel_wrap import wrap_atoms
    ideal_atoms = Atoms('Au4', [[0,0,0], [1,0,0], [0, 1, 0], [1,1,0]])
    start_atoms = Atoms('Au4', [[0,0,0], [.9,0,0], [0, .9, 0], [.9,.9,0]])

    wrap_atoms(ideal_atoms)
    wrap_atoms(start_atoms)

    gobs = wrap_pdf(ideal_atoms)[0]

    calc1 = PDFCalc(gobs=gobs, qbin=.1, conv=1000, potential='rw')
    calc2 = PDFCalc(gobs=gobs, qbin=.1, conv=1000, potential='rw')
    calc = MultiCalc(calc_list=[calc1, calc2])
    start_atoms.set_calculator(calc1)
    f = start_atoms.get_forces()
    print f
    e = start_atoms.get_total_energy()
    print e

    # view(start_atoms)