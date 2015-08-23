__author__ = 'christopher'
from ase.calculators.calculator import Calculator
import numpy as np


class Spring(Calculator):
    """
    Spring Repulsion Potential Energy Surface
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, restart=None, ignore_bad_restart_file=False, label=None,
                 atoms=None, k=10, rt=1.5, sp_type='rep', **kwargs):

        Calculator.__init__(self, restart, ignore_bad_restart_file,
                            label, atoms, **kwargs)

        if sp_type == 'com':
            from pyiid.experiments.elasticscatter.cpu_wrappers.nxn_cpu_wrap import \
                com_spring_nrg as nrg
            from pyiid.experiments.elasticscatter.cpu_wrappers.nxn_cpu_wrap import \
                com_spring_force as force
        if sp_type == 'att':
            from pyiid.experiments.elasticscatter.cpu_wrappers.nxn_cpu_wrap import \
                att_spring_nrg as nrg
            from pyiid.experiments.elasticscatter.cpu_wrappers.nxn_cpu_wrap import \
                att_spring_force as force
        self.nrg_func = nrg
        self.f_func = force
        self.sp_type = sp_type
        self.k = k
        self.rt = rt

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=['positions', 'numbers', 'cell',
                                  'pbc', 'charges', 'magmoms']):
        """Spring Calculator

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
        energy = self.nrg_func(atoms, self.k, self.rt)

        self.energy_free = energy
        self.energy_zero = energy
        self.results['energy'] = energy

    def calculate_forces(self, atoms):
        self.results['forces'] = np.zeros((len(atoms), 3))

        forces = self.f_func(atoms, self.k, self.rt)

        self.results['forces'] = forces
