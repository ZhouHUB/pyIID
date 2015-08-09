__author__ = 'christopher'
import ase.io as aseio
from ase.io.trajectory import PickleTrajectory
import matplotlib.pyplot as plt
import numpy as np
from ase.visualize import view
import matplotlib

from pyiid.wrappers.elasticscatter import ElasticScatter
from pyiid.calc import wrap_rw
from pyiid.utils import tag_surface_atoms, get_angle_list, \
    get_coord_list, get_bond_dist_list
from simdb.readers.pdfgetx3_gr import load_gr_file

from simdb.search import *
from asap3.analysis.particle import FullNeighborList, CoordinationNumbers, \
    GetLayerNumbers


def sim_unpack(sim):
    sim.reload()
    d = {}
    cl = sim.pes.calc_list
    for cal in cl:
        if cal.calculator == 'PDF':
            calc, = find_calc_document(_id=cal.id)
            d['scatter'] = calc.payload.scatter
            d['gobs'] = calc.payload.gobs
            if calc.calc_exp['ase_config_id'] is not None:
                ac, = find_atomic_config_document(
                    _id=calc.calc_exp.ase_config_id.id)
                d['target_configuration'], = ac.file_payload
    atomic_configs, = find_atomic_config_document(_id=sim.atoms.id)

    d['traj'] = atomic_configs.file_payload

    return d


if __name__ == '__main__':
    from pyiid.workflow.analysis import *

    sim = find_simulation_document(name=unicode('C60 rattle->DFT 0.05')).next()
    sim_dict = sim_unpack(sim)
    print sim_dict['target_configuration']

    ase_view(**sim_dict)
    # plot_pdf(atoms=sim_dict['traj'][-1], **sim_dict)
    # plot_pdf(atoms=sim_dict['traj'][0], **sim_dict)
    # plot_waterfall_pdf(**sim_dict)
    # plot_waterfall_diff_pdf(**sim_dict)

    # MASSIVE PROBLEM HERE DON'T KNOW WHY
    # plot_angle(1.6, **sim_dict)
    plot_coordination(1.6, **sim_dict)
