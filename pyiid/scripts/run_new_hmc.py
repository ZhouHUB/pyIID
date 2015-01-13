__author__ = 'christopher'
import cProfile

#Test total HMC
# cProfile.run(
'''
import numpy as np
from ase import Atoms
from ase.io.trajectory import PickleTrajectory
from pyiid.kernel_wrap import wrap_rw, wrap_grad_rw, wrap_pdf
from pyiid.pdfcalc import PDFCalc
from copy import deepcopy as dc
import ase.io as aseio
from pyiid.utils import load_gr_file
import matplotlib.pyplot as plt
from time import time
from pyiid.hmc import run_hmc

#load atoms and experiment
atomsi = aseio.read('/home/christopher/pdfgui_np_25_rattle1_cut.xyz')
r, pdf = load_gr_file('/home/christopher/7_7_7_FinalSum.gr')

#remove last data point, so same shape
pdf = pdf[:-1]

#destroy internal symetries from bulk model
atomsi.rattle(.05)
#set up calculator
calc = PDFCalc(gobs=pdf, qmin=2.5, conv=.0001, qbin=.1)
atomsi.set_calculator(calc)

#initialize the velocities
atomsi.set_velocities(np.zeros((len(atomsi), 3)))
pe_list = []

traj, accept_list, move_list = run_hmc(atomsi, 1, .005, 5, 0.9, 0, .9,
                                       1.02, .98, .001, .65)
'''
# , sort='percall')

#Test one force call
# cProfile.run(
'''
import numpy as np
from ase import Atoms
from ase.io.trajectory import PickleTrajectory
from pyiid.kernel_wrap import wrap_rw, wrap_grad_rw, wrap_pdf
from pyiid.pdfcalc import PDFCalc
from copy import deepcopy as dc
import ase.io as aseio
from pyiid.utils import load_gr_file
import matplotlib.pyplot as plt
from time import time
from pyiid.hmc import run_hmc

#load atoms and experiment
atomsi = aseio.read('/home/christopher/pdfgui_np_25_rattle1_cut.xyz')
r, pdf = load_gr_file('/home/christopher/7_7_7_FinalSum.gr')

#remove last data point, so same shape
pdf = pdf[:-1]

#destroy internal symetries from bulk model
atomsi.rattle(.05)
#set up calculator
calc = PDFCalc(gobs=pdf, qmin=2.5, conv=.0001, qbin=.1)
atomsi.set_calculator(calc)

#initialize the velocities
atomsi.set_velocities(np.zeros((len(atomsi), 3)))
pe_list = []

atomsi.get_forces()
atomsi.get_forces()
'''
    # , sort='tottime')


#Test simulate dynamics for 10 steps
# cProfile.run(
'''
import numpy as np
from ase import Atoms
from ase.io.trajectory import PickleTrajectory
from pyiid.kernel_wrap import wrap_rw, wrap_grad_rw, wrap_pdf
from pyiid.pdfcalc import PDFCalc
from copy import deepcopy as dc
import ase.io as aseio
from pyiid.utils import load_gr_file
import matplotlib.pyplot as plt
from time import time
from pyiid.hmc import run_hmc, simulate_dynamics

#load atoms and experiment
atomsi = aseio.read('/home/christopher/pdfgui_np_25_rattle1_cut.xyz')
r, pdf = load_gr_file('/home/christopher/7_7_7_FinalSum.gr')

#remove last data point, so same shape
pdf = pdf[:-1]

#destroy internal symetries from bulk model
atomsi.rattle(.05)
#set up calculator
calc = PDFCalc(gobs=pdf, qmin=2.5, conv=.0001, qbin=.1)
atomsi.set_calculator(calc)

#initialize the velocities
atomsi.set_velocities(np.zeros((len(atomsi), 3)))
pe_list = []
traj = simulate_dynamics(atomsi, .005, 10)
'''
    # , sort='tottime')

'''
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput

graphviz = GraphvizOutput(output_file='../extra/cpu_force'
                                      '.png')

import numpy as np
from ase import Atoms
from ase.io.trajectory import PickleTrajectory
from pyiid.kernel_wrap import wrap_rw, wrap_grad_rw, wrap_pdf
from pyiid.pdfcalc import PDFCalc
from copy import deepcopy as dc
import ase.io as aseio
from pyiid.utils import load_gr_file
import matplotlib.pyplot as plt
from time import time
from pyiid.hmc import run_hmc, simulate_dynamics

#load atoms and experiment
atomsi = aseio.read('/home/christopher/pdfgui_np_25_rattle1_cut.xyz')
r, pdf = load_gr_file('/home/christopher/7_7_7_FinalSum.gr')

#remove last data point, so same shape
pdf = pdf[:-1]

#destroy internal symetries from bulk model
atomsi.rattle(.05)
#set up calculator
calc = PDFCalc(gobs=pdf, qmin=2.5, conv=.0001, qbin=.1)
atomsi.set_calculator(calc)

#initialize the velocities
atomsi.set_velocities(np.zeros((len(atomsi), 3)))
with PyCallGraph(output=graphviz):
    atomsi.get_forces()
    '''