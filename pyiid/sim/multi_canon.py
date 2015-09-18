from __future__ import division
import numpy as np
from copy import deepcopy as dc
import importlib
from pyiid.sim import Ensemble

__author__ = 'christopher'


class MultiCanonicalSimulation(Ensemble):
    def __init__(self, atoms, ensemble_list,
                 logfile=None, trajectory=None, ensemble_prob=None,
                 seed=None, verbose=False):
        Ensemble.__init__(self, atoms, logfile, trajectory, seed, verbose)
        self.ensembles = []

        # build the ensemble and init
        for ensemble in ensemble_list:
            self.ensembles.append(ensemble)

        if ensemble_prob is not None:
            self.prob = ensemble_prob
        else:
            self.prob = None

    def step(self):
        j = np.random.choice(np.arange(len(self.ensembles)), p=self.prob)
        new_configs = self.ensembles[j].step()
        if new_configs:
            self.traj.extend(new_configs)
            for i in range(len(self.ensembles)):
                if i != j:
                    self.ensembles[i].traj.extend(new_configs)
