from __future__ import division
import numpy as np
from copy import deepcopy as dc
import importlib
from pyiid.sim import Ensemble
__author__ = 'christopher'

supported_ensebles = {
    'NUTS': ['pyiid.sim.nuts_hmc', 'NUTSMove'],
    'GCMC': ['pyiid.sim.gcmc', 'GCMove']
}


class MultiCanonicalSimulation(Ensemble):
    def __init__(self, atoms, ensemble_dict,
                 logfile, trajectory, ensemble_prob=None):
        Ensemble.__init__(self, atoms, logfile, trajectory)
        self.ensembles = []

        # build the ensemble and init
        for ensemble, value in ensemble_dict.items():
            if ensemble in supported_ensebles.keys():
                mod = importlib.import_module(supported_ensebles[ensemble][0])
                e = getattr(mod, supported_ensebles[ensemble][1])
                self.ensembles.append(e(**value))

        if ensemble_prob is not None:
            self.prob = ensemble_prob
        else:
            self.prob = None

    def step(self):
        j = np.random.choice(np.arange(len(self.ensembles)), p=self.prob)
        self.ensembles[j].step()
