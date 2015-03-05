__author__ = 'christopher'
# cProfile.run('''
from copy import deepcopy as dc

from pyiid.wrappers.gpu_wrap import *


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
