__author__ = 'christopher'
from ase.calculators.calculator import Calculator
import numpy as np
from copy import deepcopy as dc


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
        self.list_atoms = []


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
        if not self.list_atoms:
            self.list_atoms = [dc(atoms) for i in self.calc_list]
            for ele, calc in zip(self.list_atoms, self.calc_list):
                ele.set_calculator(calc)
        else:
            for ele in self.list_atoms:
                ele.positions = atoms.positions
        energy=0
        for ele in self.list_atoms:
            energy += ele.get_potential_energy()
        # energy = sum([ele.get_potential_energy() for ele in self.atoms_list])
        self.energy_free = energy
        self.energy_zero = energy
        self.results['energy'] = energy

    def calculate_forces(self, atoms):
        self.results['forces'] = np.zeros((len(atoms), 3))
        forces = np.zeros((len(atoms), 3))

        if not self.list_atoms:
            self.list_atoms = [dc(atoms) for i in self.calc_list]
            for ele, calc in zip(self.list_atoms, self.calc_list):
                ele.set_calculator(calc)
        else:
            for ele in self.list_atoms:
                ele.positions = atoms.positions
        for ele in self.list_atoms:
            forces[:,:] += ele.get_forces()

        self.results['forces'] = forces


if __name__ == '__main__':
    from ase.atoms import Atoms
    from ase.visualize import view
    from pyiid.calc.pdfcalc import PDFCalc
    from pyiid.calc.spring_calc import Spring
    from pyiid.wrappers.elasticscatter import wrap_atoms, ElasticScatter
    from ase.cluster.octahedron import Octahedron
    from ase.calculators.lammpslib import LAMMPSlib
    import numpy as np
    import matplotlib.pyplot as plt

    # ideal_atoms = Atoms('Au4', [[0,0,0], [3,0,0], [0, 3, 0], [3,3,0]])
    # start_atoms = Atoms('Au4', [[0,0,0], [.9,0,0], [0, .9, 0], [.9,.9,0]])
    ideal_atoms = Octahedron('Au', 2)
    ideal_atoms.pbc = False
    s = ElasticScatter()

    gobs = s.get_pdf(ideal_atoms)

    pair_style = 'eam'
    Au_eam_file = '/mnt/work-data/dev/IID_data/examples/Au/736_atom/Au_sheng.eam'

    cmds = ["pair_style eam/alloy",
            "pair_coeff * * " + str(Au_eam_file) + " " + "Au"]

    calc2 = LAMMPSlib(lmpcmds=cmds, logfile='test.log')
    super_el = []
    js = np.linspace(0, 150, 10)
    for j in js:
        print j
        calc1 = PDFCalc(gobs=gobs, qbin=.1, conv=j, potential='rw')
        calc_list = [
            calc1,
            calc2
        ]
        el = []
        calc = MultiCalc(calc_list=calc_list)
        ideal_atoms.set_calculator(calc)
        traj = []
        for i in np.linspace(.75, 1.25, 101):
            atoms = dc(ideal_atoms)
            atoms.positions *= i
            e = atoms.get_potential_energy()
            el.append(e)
            e2 = 2 * e
            f = atoms.get_forces()
            f2 = 2 * f
            traj += [atoms]
        super_el.append(el)
        view(traj)
        # plt.plot(np.linspace(.75, 1.25, 101), el, label=str(j))
        # plt.legend()
        # plt.show()
    for el, conv in zip(super_el, js):
        plt.plot(np.linspace(.75, 1.25, 101)*ideal_atoms.get_distance(0, 1), el/conv, label=conv)
    plt.ylabel('normalized energy')
    plt.xlabel('first bond distance')
    plt.legend(title='PDF conversion factor')
    plt.title('Au Octahedron expansion PES')
    plt.show()