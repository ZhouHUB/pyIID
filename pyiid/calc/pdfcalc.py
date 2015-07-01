from pyiid.calc import wrap_rw, wrap_chi_sq, wrap_grad_rw, wrap_grad_chi_sq

__author__ = 'christopher'
from ase.calculators.calculator import Calculator
from pyiid.wrappers.elasticscatter import ElasticScatter
import math
import numpy as np


class PDFCalc(Calculator):
    """
    Class for doing PDF based RW/chi**2 calculations
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, restart=None, ignore_bad_restart_file=False, label=None,
                 atoms=None,
                 obs_data=None, scatter=None, exp_dict=None,
                 conv=1., potential='rw', **kwargs):

        Calculator.__init__(self, restart, ignore_bad_restart_file,
                            label, atoms, **kwargs)
        self.scale = 1
        if scatter is None:
            self.scatter = ElasticScatter(exp_dict)
        else:
            self.scatter = scatter
        self.exp_dict = self.scatter.exp
        self.rw_to_eV = conv

        if obs_data is not None and exp_dict is not None:
            if len(np.arange(exp_dict['rmin'], exp_dict['rmax'],
                             exp_dict['rstep'])) < len(obs_data):
                obs_data = obs_data[math.floor(self.exp_dict['rmin'] / self.exp_dict['rstep']):math.ceil(self.exp_dict['rmax'] / self.exp_dict['rstep'])]
            self.gobs = obs_data
        elif obs_data is not None:
            self.gobs = obs_data
        else:
            raise NotImplementedError('Need an experimental PDF')

        if potential == 'chi_sq':
            self.potential = wrap_chi_sq
            self.grad = wrap_grad_chi_sq
        elif potential == 'rw':
            self.potential = wrap_rw
            self.grad = wrap_grad_rw
        else:
            raise NotImplementedError('Potential not implemented')

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
        energy, scale = self.potential(self.scatter.get_pdf(atoms), self.gobs)
        self.scale = scale
        self.energy_free = energy * self.rw_to_eV
        self.energy_zero = energy * self.rw_to_eV

        self.results['energy'] = energy * self.rw_to_eV

    def calculate_forces(self, atoms):
        # self.results['forces'] = np.zeros((len(atoms), 3))
        forces = self.grad(self.scatter.get_grad_pdf(atoms),
                           self.scatter.get_pdf(atoms),
                           self.gobs) * self.rw_to_eV

        self.results['forces'] = forces  # * atoms.get_masses().reshape(-1, 1)
