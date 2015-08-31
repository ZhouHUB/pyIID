from __future__ import division
__author__ = 'christopher'
import numpy as np
from copy import deepcopy as dc
import importlib

supported_ensebles = {
    'NUTS': ['pyiid.sim.nuts_hmc', 'NUTSMove'],
    'GCMC': ['pyiid.sim.gcmc', 'GCMove']
}


class MultiCanonicalSimulation:
    def __init__(self, atoms, ensemble_dict, iterations, ensemble_prob=None):
        self.starting_atoms = dc(atoms)
        self.traj = [dc(atoms)]
        self.ensembles = []

        # build the ensemble and init
        for ensemble in ensemble_dict.keys():
            if ensemble in supported_ensebles.keys():
                mod = importlib.import_module(supported_ensebles[ensemble][0])
                e = getattr(mod, supported_ensebles[ensemble][1])
                self.ensembles.append(e(**ensemble_dict[ensemble]))

        self.total_iterations = iterations
        if ensemble_prob is not None:
            self.prob = ensemble_prob
        else:
            self.prob = np.ones(len(self.ensembles))/len(self.ensembles)

    def run(self):
        for i in xrange(self.total_iterations):
            self.step()

    def step(self):
        j = np.random.choice(np.range(len(self.ensembles)), p=self.prob)
        self.ensembles[j].run()
