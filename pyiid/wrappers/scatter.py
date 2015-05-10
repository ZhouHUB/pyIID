__author__ = 'christopher'
from numba import cuda
import numpy as np
import math

from pyiid.wrappers.cpu_wrap import wrap_fq as cpu_wrap_fq
from pyiid.wrappers.cpu_wrap import wrap_fq_grad as cpu_wrap_fq_grad

from pyiid.wrappers.multi_gpu_wrap import wrap_fq as node_0_gpu_wrap_fq
from pyiid.wrappers.multi_gpu_wrap import \
    wrap_fq_grad as node_0_gpu_wrap_fq_grad

from pyiid.wrappers.mpi_gpu_wrap import wrap_fq as multi_node_gpu_wrap_fq
from pyiid.wrappers.mpi_gpu_wrap import \
    wrap_fq_grad as multi_node_gpu_wrap_fq_grad

from pyiid.kernels.master_kernel import get_pdf_at_qmin, \
    get_rw, grad_pdf, get_grad_rw, get_chi_sq, get_grad_chi_sq


def check_mpi():
    # Test if MPI GPU is viable
    # Currently no working MPI GPU implementation
    return False


def check_multi_gpu():
    try:
        cuda.get_current_device()
        cuda.close()
        return True
    except:
        return False


class Scatter(object):
    """
    Scatter contains all the methods associated with producing theoretical
    diffraction patterns and PDFs from atomic configurations.  It does not
    include potential energies, such as Rw and chi**2 which are under the
    Calculator object.
    """

    def __init__(self, exp_dict=None):
        # Currently supported processor architectures
        self.avail_pro = ['MPI-GPU', 'Multi-GPU', 'Serial-CPU']

        # Should be read in from the gr file, but if not here are some defaults
        if exp_dict is None or bool(exp_dict) is False:
            exp_dict = {'qmin': 0.0, 'qmax': 25., 'qbin': .1, 'rmin': 0.0,
                        'rmax': 40.0, 'rstep': .01}

        # Technically we should use this for qbin:
        # self.qbin = np.pi / (rmax + 6 * 2 * np.pi / qmax)

        self.exp = exp_dict

        # Just in case something blows up down the line
        self.fq = cpu_wrap_fq
        self.grad = cpu_wrap_fq_grad
        self.processor = 'Serial-CPU'

        # Get the fastest processor architecture available
        self.set_processor()

    def set_processor(self, processor=None):
        """
        Set the processor to use for calculating the scattering.  If no
        parameter is given then check for the fastest possible processor
        configuration

        Parameters
        -----------
        processor: ['MPI-GPU', 'Multi-GPU', 'Serial-CPU']
            The processor to use
        """
        # If a processor is given try to use that processor, but check if it is
        # viable first.

        if processor is None:
            # Test each processor in order of most advanced to least
            for pro in self.avail_pro:
                if self.set_processor(processor=pro) is not None:
                    break

        elif processor == self.avail_pro[0] and check_mpi() is True:
            self.fq = multi_node_gpu_wrap_fq
            self.grad = multi_node_gpu_wrap_fq_grad
            self.processor = processor

        elif processor == self.avail_pro[1] and check_multi_gpu() is True:
            self.fq = node_0_gpu_wrap_fq
            self.grad = node_0_gpu_wrap_fq_grad
            self.processor = processor

        elif processor == self.avail_pro[2]:
            self.fq = cpu_wrap_fq
            self.grad = cpu_wrap_fq_grad
            self.processor = processor

    def get_fq(self, atoms):
        return self.fq(atoms, self.exp['qmax'], self.exp['qbin'])

    def get_pdf(self, atoms):
        fq = self.fq(atoms)
        fq[:int(self.exp['qmin'] / self.exp['qbin'])] = 0
        pdf0 = get_pdf_at_qmin(fq, self.exp['rstep'], self.exp['qbin'],
                               np.arange(0, self.exp['rmax'],
                                         self.exp['rstep']))
        pdf = pdf0[int(self.exp['rmin'] / self.exp['rstep']):int(
            self.exp['rmax'] / self.exp['rstep'])]
        return pdf

    def get_sq(self, atoms):
        fq = self.fq(atoms)
        scatter_vector = np.arange(0, self.exp['qmax'], self.exp['qbin'])
        print fq.shape, scatter_vector.shape
        sq = (fq / scatter_vector) + np.ones(scatter_vector.shape)
        sq[np.isinf(sq)] = 0
        return sq

    def get_iq(self, atoms):
        return self.get_sq(atoms) * np.average(atoms.get_array('scatter')) ** 2

    def get_grad_fq(self, atoms):
        return self.grad(atoms, self.exp['qmax'], self.exp['qbin'])

    def get_grad_pdf(self, atoms):
        fq_grad = self.grad(atoms)
        qmin_bin = int(self.exp['qmin'] / self.exp['qbin'])
        for tx in range(len(atoms)):
            for tz in range(3):
                fq_grad[tx, tz, :qmin_bin] = 0.
        pdf_grad = np.zeros(
            (len(atoms), 3, self.exp['rmax'] / self.exp['rstep']))
        grad_pdf(pdf_grad, fq_grad, self.exp['rstep'], self.exp['qbin'],
                 np.arange(0, self.exp['rmax'], self.exp['rstep']))
        return pdf_grad[:, :,
               math.floor(self.exp['rmin'] / self.exp['rstep']):]


if __name__ == '__main__':
    from ase.atoms import Atoms
    from pyiid.wrappers.master_wrap import wrap_atoms
    import matplotlib.pyplot as plt

    atoms = Atoms('Au4', [[0, 0, 0], [3, 0, 0], [0, 3, 0], [3, 3, 0]])
    wrap_atoms(atoms)

    scat = Scatter()
    # scat.set_processor('Serial-CPU')
    scat.set_processor()
    print scat.processor
    print scat.get_grad_fq(atoms)
    plt.plot(scat.get_iq(atoms))
    plt.show()