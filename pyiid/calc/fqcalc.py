from ase.calculators.calculator import Calculator
from pyiid.calc import wrap_rw, wrap_chi_sq, wrap_grad_rw, wrap_grad_chi_sq
from pyiid.experiments.elasticscatter import ElasticScatter

__author__ = 'christopher'


class FQCalc(Calculator):
    """
    Class for doing FQ PES calculations
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, restart=None, ignore_bad_restart_file=False, label=None,
                 atoms=None, obs_data=None, scatter=ElasticScatter(), conv=1.,
                 potential='chi_sq', **kwargs):

        Calculator.__init__(self, restart, ignore_bad_restart_file,
                            label, atoms, **kwargs)
        self.scatter = scatter
        self.rw_to_eV = conv

        if obs_data is not None:
            self.gobs = obs_data
        else:
            raise NotImplementedError('Need an experimental F(Q)')

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
        energy, scale = self.potential(self.scatter.get_fq(atoms), self.gobs)
        self.results['energy'] = energy * self.rw_to_eV

    def calculate_forces(self, atoms):
        forces = self.grad(self.scatter.get_grad_fq(atoms),
                           self.scatter.get_fq(atoms),
                           self.gobs) * self.rw_to_eV

        self.results['forces'] = forces
