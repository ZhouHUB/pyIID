__author__ = 'christopher'
import ase.io as aseio
from ase.io.trajectory import PickleTrajectory
import matplotlib.pyplot as plt
import numpy as np

from pyiid.wrappers.scatter import Scatter
from pyiid.calc.oo_pdfcalc import wrap_rw
from pyiid.utils import load_gr_file, tag_surface_atoms, get_angle_list



# We would like to have a way to setup and run the simulation, collecting
# metadata along the way, analyze the results, keep track of comments, and
# where all the files went/are going.  This should be loadable, giving us the
# ability to restart simulations, run the same simulation over again, and
# quickly grab the analysis figures/data.
# In short a proto-DB, possibly written in json.


def plot_pdf(db_entry, save_file=None, show=True, sl='last'):
    scatter = Scatter(db_entry['exp_dict'])
    if db_entry['exp_type'] == 'theory':
        ideal_atoms = aseio.read(str(db_entry['exp_files']))
        gobs = scatter.get_pdf(ideal_atoms)
    else:
        r, gobs, exp_dict = load_gr_file(str(db_entry['exp_files']))

    traj = PickleTrajectory(db_entry['traj loc'])
    start_atoms = traj[0]
    print 'Start Rw', wrap_rw(scatter.get_pdf(start_atoms), gobs)[0] * 100, '%'

    if sl == 'last':
        atoms = traj[-1]
    elif type(sl) is int:
        atoms = traj[sl]
    elif sl =='all':
        atoms = traj

    gcalc = scatter.get_pdf(atoms)
    r = scatter.get_r()


    rw, scale = wrap_rw(gcalc, gobs)
    print 'Final Rw', rw * 100, '%'

    baseline = -1 * np.abs(1.5 * gobs.min())
    gdiff = gobs - gcalc * scale

    plt.figure()
    plt.plot(r, gobs, 'bo', label="G(r) data")
    plt.plot(r, gcalc * scale, 'r-', label="G(r) fit")
    plt.plot(r, gdiff + baseline, 'g-', label="G(r) diff")
    plt.plot(r, np.zeros_like(r) + baseline, 'k:')
    plt.xlabel(r"$r (\AA)$")
    plt.ylabel(r"$G (\AA^{-2})$")
    plt.legend()
    if save_file is not None:
        plt.savefig(save_file + '_pdf.eps', bbox_inches='tight',
                    transparent='True')
        plt.savefig(save_file + '_pdf.png', bbox_inches='tight',
                    transparent='True')
    if show is True:
        plt.show()
    return


def plot_angle(db_entry, cut, save_file=None, show=True):
    traj = PickleTrajectory(str(db_entry['traj loc']), 'r')
    stru_l = {}
    if db_entry['exp_type'] == 'theory':
        stru_l['Target'] = aseio.read(str(db_entry['exp_files']))

    stru_l['Start'] = traj[0]
    stru_l['Finish'] = traj[-1]
    for atoms in stru_l.values():
        tag_surface_atoms(atoms, cut)

    symbols = set(stru_l['Start'].get_chemical_symbols())
    tags = {'Core': (0, '+'), 'Surface': (1, '*')}
    for tag in tags.keys():
        tagged_atoms = stru_l['Start'][[atom.index for atom in stru_l['Start'] if atom.tag == tags[tag][0]]]
        if len(tagged_atoms) == 0:
            del tags[tag]
    colors = ['c', 'm', 'y', 'k']
    bins = np.linspace(0, 180, 100)
    # Bin the data
    for n, key in enumerate(stru_l.keys()):
        for symbol in symbols:
            for tag in tags.keys():
                a, b = np.histogram(get_angle_list(stru_l[key], cut, element=symbol,
                                      tag=tags[tag][0]), bins=bins)
                if np.alltrue(stru_l[key].pbc):
                    # crystal
                    for y, x in zip(a, b[:-1]):
                        plt.axvline(x=x, ymax=y, color='grey', linestyle='--')
                else:
                    plt.plot(b[:-1], a, label=key+' '+symbol+' '+tag,
                             marker=tags[tag][1], color=colors[n])
    plt.xlabel('Bond angle in Degrees')
    plt.xlim(0, 180)
    plt.ylabel('Angle Counts')
    plt.legend()
    if save_file is not None:
        plt.savefig(save_file + '_angle.eps', bbox_inches='tight',
                    transparent='True')
        plt.savefig(save_file + '_angle.png', bbox_inches='tight',
                    transparent='True')
    if show is True:
        plt.show()

if __name__ == '__main__':
    from pyiid.workflow.db_utils import load_db
    db = load_db('/mnt/work-data/dev/IID_data/db_test/test.json')
    plot_angle(db[-1], 3)
