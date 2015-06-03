__author__ = 'christopher'
import sys
import inspect
from threading import Thread

import pyiid.wrappers.mpi.mpi_gpu_avail as mpi_gpu_avail
import pyiid.wrappers.mpi.mpi_fq_worker as mpi_fq_worker
import pyiid.wrappers.mpi.mpi_grad_worker as mpi_grad_worker


def gpu_avail(n_nodes):
    from mpi4py import MPI
    avail_loc = inspect.getfile(mpi_gpu_avail)
    comm = MPI.COMM_WORLD.Spawn(sys.executable,
        args=[avail_loc],
        maxprocs=n_nodes
    )
    ranks = comm.gather(root=MPI.ROOT)
    mem_list = comm.gather(root=MPI.ROOT)
    comm.Disconnect()
    return ranks, mem_list


def mpi_fq(n_nodes, m_list, q, scatter_array, qbin):
    from mpi4py import MPI
    kernel_loc = inspect.getfile(mpi_fq_worker)
    comm = MPI.COMM_WORLD.Spawn(
        sys.executable,
        args=[kernel_loc],
        maxprocs=n_nodes
    )
    n_cov = 0
    status = MPI.Status()
    m_list += ([StopIteration] * n_nodes)
    p = None
    for m in m_list:
        if m is StopIteration:
            msg = m
        else:
            msg = (q, scatter_array, qbin, m, n_cov)
        # If the thread on the main node is done, or not started give a problem to it
        if p is None or p.is_alive() is False:
            p = Thread(target=None, args=msg)
        comm.recv(source=MPI.ANY_SOURCE, status=status)
        comm.send(obj=msg, dest=status.Get_source())
    # print 'done master'
    reports = comm.gather(root=MPI.ROOT)
    comm.Disconnect()
    return reports


def mpi_grad_fq(n_nodes, m_list, q, scatter_array, qmax_bin, qbin):
    from mpi4py import MPI
    kernel_loc = inspect.getfile(mpi_grad_worker)
    comm = MPI.COMM_WORLD.Spawn(
        sys.executable,
        args=[kernel_loc],
        maxprocs=n_nodes
    )
    n_cov = 0
    status = MPI.Status()
    m_list += ([StopIteration] * n_nodes)
    for m in m_list:
        if m is StopIteration:
            msg = m
        else:
            msg = (q, scatter_array, qbin, m, n_cov)
        comm.recv(source=MPI.ANY_SOURCE, status=status)
        comm.send(obj=msg, dest=status.Get_source())

    reports = comm.gather(root=MPI.ROOT)
    return reports
