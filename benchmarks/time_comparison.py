__author__ = 'christopher'
import pickle
import traceback

from ase.atoms import Atoms
import matplotlib.pyplot as plt

from pyiid.calc.calc_1d import Calc1D
from pyiid.utils import build_sphere_np
import time
from pyiid.experiments.elasticscatter import ElasticScatter

exp = None
scat = ElasticScatter()
atoms = Atoms('Au4', [[0,0,0], [3,0,0], [0,3,0], [3,3,0]])
pdf = scat.get_pdf(atoms)

type_list = []
time_list = []
benchmarks = [
    # ('CPU', 'flat'),
    ('Multi-GPU', 'flat')
]
colors=['b', 'r']
sizes = range(10, 55, 5)
print sizes

for proc, alg in benchmarks:
    scat.set_processor(proc, alg)
    type_list.append((proc, alg))
    nrg_l = []
    f_l = []
    try:
        for i in sizes:
            atoms = build_sphere_np('/mnt/work-data/dev/pyIID/benchmarks/1100138.cif', float(i) / 2)
            atoms.rattle()
            print len(atoms), i/10.
            calc = Calc1D(target_data=pdf,
                          exp_function=scat.get_fq,
                          exp_grad_function=scat.get_grad_fq,
                          conv=1, potential='rw')
            atoms.set_calculator(calc)

            s = time.time()
            nrg = atoms.get_potential_energy()
            # scat.get_fq(atoms)
            f = time.time()

            nrg_l.append(f-s)

            s = time.time()
            # force = atoms.get_forces()
            # scat.get_grad_fq(atoms)
            f = time.time()
            f_l.append(f-s)
    except:
        print traceback.format_exc()
        break
    time_list.append((nrg_l, f_l))

for i in range(len(benchmarks)):
    for j, calc_type in enumerate(['energy', 'force']):
        f_str = benchmarks[i][0]+'_'+benchmarks[i][0]+'_'+calc_type+'.pkl'
        with open(f_str, 'w') as f:
            pickle.dump(time_list[i][j], f)
'''
for i in range(len(benchmarks)):
    for j, (calc_type, line) in enumerate(zip(['energy', 'force'], ['o', 's'])):
        plt.plot(sizes,time_list[i][j], color=colors[i], marker=line, label= '{0} {1}'.format(benchmarks[i][0], calc_type))
plt.legend(loc='best')
plt.xlabel('NP diameter in Angstrom')
plt.ylabel('time (s) [lower is better]')
plt.savefig('speed3.eps', bbox_inches='tight', transparent=True)
plt.savefig('speed3.png', bbox_inches='tight', transparent=True)
plt.show()
'''

for i in range(len(benchmarks)):
    for j, (calc_type, line) in enumerate(zip(['energy', 'force'], ['o', 's'])):
        plt.semilogy(sizes,time_list[i][j], color=colors[i], marker=line, label= '{0} {1}'.format(benchmarks[i][0], calc_type))
plt.legend(loc='best')
plt.xlabel('NP diameter in Angstrom')
plt.ylabel('time (s) [lower is better]')
# plt.savefig('speed_log3.eps', bbox_inches='tight', transparent=True)
# plt.savefig('speed_log3.png', bbox_inches='tight', transparent=True)
plt.show()