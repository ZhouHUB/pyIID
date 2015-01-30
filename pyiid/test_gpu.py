__author__ = 'christopher'
import cProfile
# cProfile.run('''
from ase.atoms import Atoms as atoms
import ase.io as aseio
from ase.visualize import view
import matplotlib.pyplot as plt
from timeit import default_timer as time
import numpy as np
from numbapro import cuda
from copy import deepcopy as dc
from pyiid.wrappers.gpu_wrap import *
from pyiid.serial_kernel import get_d_array, get_r_array
import math

atomsio = aseio.read(
    '/mnt/bulk-data/Dropbox/BNL_Project/Simulations/Models.d/1-C60.d/C60.xyz')
scatter_array = np.loadtxt('/mnt/bulk-data/Dropbox/BNL_Project/Simulations/Models.d/1-C60.d/c60_scat.txt', dtype=np.float32)
atomsio.set_array('scatter', scatter_array)
atoms = dc(atomsio)
# for i in range(10):
#     atoms += atomsio
print scatter_array.shape

s= time()
pdf, fq = wrap_pdf_gpu(atoms)
print(time()-s)
plt.plot(pdf)
plt.show()
