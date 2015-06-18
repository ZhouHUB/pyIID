__author__ = 'christopher'
import ase.io as aseio
from ase.io.trajectory import PickleTrajectory
import matplotlib.pyplot as plt
import numpy as np
from ase.visualize import view
import matplotlib

from pyiid.wrappers.elasticscatter import ElasticScatter
from pyiid.calc import wrap_rw
from pyiid.utils import load_gr_file, tag_surface_atoms, get_angle_list, \
    get_coord_list, get_bond_dist_list


font = {'family': 'normal',
        # 'weight' : 'bold',
        'size': 18}

matplotlib.rc('font', **font)
plt.ion()


# We would like to have a way to setup and run the simulation, collecting
# metadata along the way, analyze the results, keep track of comments, and
# where all the files went/are going.  This should be loadable, giving us the
# ability to restart simulations, run the same simulation over again, and
# quickly grab the analysis figures/data.
# In short a proto-DB, possibly written in json.


def plot_pdf(db_entry, save_file=None, show=True, sl='last'):
    scatter = ElasticScatter(db_entry['exp_dict'])
    if db_entry['exp_type'] == 'theory':
        ideal_atoms = aseio.read(str(db_entry['exp_files']))
        gobs = scatter.get_pdf(ideal_atoms)
    else:
        r, gobs, exp_dict = load_gr_file(str(db_entry['exp_files']), **db_entry['exp_dict'])

    traj = PickleTrajectory(db_entry['traj loc'])
    start_atoms = traj[0]
    print 'Start Rw', wrap_rw(scatter.get_pdf(start_atoms), gobs)[0] * 100, '%'

    if sl == 'last':
        atoms = traj[-1]
    elif type(sl) is int:
        atoms = traj[sl]
    elif sl == 'all':
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
        tagged_atoms = stru_l['Start'][
            [atom.index for atom in stru_l['Start'] if
             atom.tag == tags[tag][0]]]
        if len(tagged_atoms) == 0:
            del tags[tag]
    colors = ['c', 'm', 'y', 'k']
    bins = np.linspace(0, 180, 100)
    # Bin the data
    for n, key in enumerate(stru_l.keys()):
        for symbol in symbols:
            for tag in tags.keys():
                a, b = np.histogram(
                    get_angle_list(stru_l[key], cut, element=symbol,
                                   tag=tags[tag][0]), bins=bins)
                if False:
                    pass
                # if np.alltrue(stru_l[key].pbc):
                    # crystal
                    # for y, x in zip(a, b[:-1]):
                    #     plt.axvline(x=x, ymax=y, color='grey', linestyle='--')
                else:
                    plt.plot(b[:-1], a, label=key + ' ' + symbol + ' ' + tag,
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


def plot_coordination(db_entry, cut, save_file=None, show=True):
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
        tagged_atoms = stru_l['Start'][
            [atom.index for atom in stru_l['Start'] if
             atom.tag == tags[tag][0]]]
        if len(tagged_atoms) == 0:
            del tags[tag]
    b_min = None
    b_max = None
    for key in stru_l.keys():
        total_coordination = get_coord_list(stru_l[key], cut)
        l_min = min(total_coordination)
        l_max = max(total_coordination)
        if b_min is None or b_min > l_min:
            b_min = l_min
        if b_max is None or b_max < l_max:
            b_max = l_max
    if b_min == b_max:
        bins = np.asarray([b_min, b_max])
    else:
        bins = np.arange(b_min, b_max+2)
    print bins
    width = 3. / 4 / len(stru_l)
    offset = .3 * 3 / len(stru_l)
    patterns = ('x', '\\', 'o', '.')
    colors = ['grey', 'mediumseagreen', 'c', 'y', 'red', 'blue']
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for n, key in enumerate(stru_l.keys()):
        bottoms = np.zeros(bins.shape)
        j = 0
        for symbol in symbols:
            for tag in tags.keys():
                hatch = patterns[j]
                coord = get_coord_list(stru_l[key], cut, element=symbol,
                                       tag=tags[tag][0])
                print coord
                a, b = np.histogram(coord, bins=bins)
                print b[:-1]
                ax.bar(b[:-1] + n * offset, a, width, bottom=bottoms[:-1],
                       color=colors[n], label=key + ' ' + symbol + ' ' + tag,
                       hatch=hatch)
                j += 1
                bottoms[:-1] += a

    plt.xlabel('Coordination Number')
    plt.xticks(bins[:-1] + 1 / 2., bins[:-1])
    plt.ylabel('Atomic Counts')
    ax2 = plt.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax.legend(loc=0, ncol=1)
    if save_file is not None:
        plt.savefig(save_file + '_coord.eps', bbox_inches='tight',
                    transparent='True')
        plt.savefig(save_file + '_coord.png', bbox_inches='tight',
                    transparent='True')
    if show is True:
        plt.show()
    return


def plot_bonds(db_entry, cut, save_file=None, show=True):
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
        tagged_atoms = stru_l['Start'][
            [atom.index for atom in stru_l['Start'] if
             atom.tag == tags[tag][0]]]
        if len(tagged_atoms) == 0:
            del tags[tag]
    colors = ['c', 'm', 'y', 'k']
    for n, key in enumerate(stru_l.keys()):
        for symbol in symbols:
            for tag in tags.keys():
                bonds = get_bond_dist_list(stru_l[key], cut, element=symbol,
                                   tag=tags[tag][0])
                a, b = np.histogram(bonds, bins=10)
                plt.plot(b[:-1], a, label=key + ' ' + symbol + ' ' + tag,
                             marker=tags[tag][1], color=colors[n])
    plt.xlabel('Bond distance in angstrom')
    plt.ylabel('Bond Counts')
    plt.legend(loc='best')
    if save_file is not None:
        plt.savefig(save_file + '_angle.eps', bbox_inches='tight',
                    transparent='True')
        plt.savefig(save_file + '_angle.png', bbox_inches='tight',
                    transparent='True')
    if show is True:
        plt.show()


def ase_view(db_entry):
    traj = PickleTrajectory(db_entry['traj loc'], 'r')[:]
    view(traj)


if __name__ == '__main__':
    from pyiid.workflow.db_utils import load_db

    db = load_db('/mnt/work-data/dev/IID_data/db_test/test.json')
    # plot_angle(db[-1], 3)
    # plot_coordination(db[-1], 3)
    # plot_coordination(db[-4], 1.6)
    plot_bonds(db[-2], 3)