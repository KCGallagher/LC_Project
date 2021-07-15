"""Optimised method to calculate the pair-wise orientational order correlation function.
As detailed by Daan Frenkel, uses spherical harmonics so avoid costly angle calculations (which are O(N^2))
as well as fourier transform methods. 
Uses the end-to-end molecule vector as the director."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from scipy.ndimage import uniform_filter1d  # for rolling average
from scipy.special import sph_harm
from phase_plot import vol_frac

FILE_ROOT = "output_T_0.5_time_"  # two underscores to match typo in previous code
SAMPLING_FREQ = 20  # only samples one in X files (must be integer)
SEPARATION_BIN_NUM = 20  # number of bins for radius dependance pair-wise correlation

# mol_length = 10  #uncomment on older datasets

plt.rcParams.update({"font.size": 13})  # for figures to go into latex at halfwidth

# READ PARAMETER VALUES FROM LOG FILE

file_name = "log.lammps"
log_file = open(file_name, "r")
mix_steps_values = []

for i, line in enumerate(log_file):
    """For loop iteratres over every line in file to find the required variables.

    However, because the dump_interval occurs last (in the main body not the preamble)
    we break at this point to avoid reading the whole file unnecessarily."""

    if "variable N" in line:  # to extract independant variable value of N
        for t in line.split():  # separate by whitespace
            try:
                N_molecules = int(t)
            except ValueError:
                pass  # any non-floats in this line are ignored

    if "variable len" in line:  # to extract length of molecule
        for t in line.split():
            try:
                mol_length = int(t)
            except ValueError:
                pass  # any non-floats in this line are ignored

    if "dump" and "all custom" in line:  # to extract time interval for dump
        for t in line.split():
            try:
                dump_interval = int(t)
                # interval is the last integer in that line
            except ValueError:
                pass
        break  # got all data, break from for loop

    if "variable mix_steps" in line:  # to extract mix_steps (for shrinking)
        run_num = 0  # counts number of runs
        for t in line.split():
            try:
                mix_steps_values.append(int(t))
                run_num += 1
            except ValueError:
                pass

    if "variable run_steps" in line:  # to extract run_steps (for equilibration)
        for t in line.split():
            try:
                equilibrium_time = int(t)  # time per run
            except ValueError:
                pass
        # this comes up first in file so no searching variable here

log_file.close()

tot_mix_time = sum(mix_steps_values)
run_time = tot_mix_time + run_num * equilibrium_time
time_range = range(0, int(run_time), int(dump_interval * SAMPLING_FREQ))
print(
    "N_molecules, run_time, dump_interval = "
    + str((N_molecules, run_time, dump_interval))
)


# time_range = range(0, 3300000, 100000)  # FOR SIMPLICITY IN TESTING


def find_angles(vec_array):
    """Finds spherical angles of cartesian position. Returns theta and phi in N x 2 array
    
    vec_array should be of size N x 3, for N angles (ie N particles)
    Choice of (normalised) base vector is arbitrary in order param formulation"""

    assert vec_array.shape[1] == 3, "Vectors should be three dimensional"
    radius = np.linalg.norm(vec_array, axis=1)
    theta = np.arccos(vec_array[:, 2] / radius)
    phi = np.arctan2(vec_array[:, 1], vec_array[:, 0])
    return theta, phi


def find_separations(pos_array, base_pos, box_dim):
    """Finds separation of positions in array from a base position

    This method finds the minimum separation, accounting for the periodic BC
    pos1, pos2 are the position vectors of the two points
    box_dim is a vector (of equal length) giving the dimensions of the simulation region"""

    separation_data = pos_array - base_pos
    for n in range(pos_array.shape[0]):
        for i in range(3):  # for 3 dimensional vector
            if np.abs(pos_array[n, i] - base_pos[i]) > box_dim[i] / 2:
                # use distance to ghost instead
                separation_data[n, i] = box_dim[i] - np.abs(
                    pos_array[n, i] - base_pos[i]
                )

    return np.linalg.norm(separation_data, axis=1)


def spherical_harmonic_sum(theta_array, phi_array, l, m):
    """ Returns sum  of spherical harmonics of order l,m"""
    angles_sum = np.sum(sph_harm(m, l, theta_array, phi_array).real)
    return angles_sum


def correlation_func(data, box_dim, bin_num, order):
    """Input data in array of size Molecule Number x 3 x 3

    Input data will be rod_positions array which stores input data   
    First index gives molecule number (up to size N_molecules)
    Second index gives particle number within molecule (first/last)
    Third index gives the component of the position (x,y,z)

    Also input  'order' - order of correlation function to compute
                'box_dim' - list of simulation box dimensions
                'bin_num' - integer value for number of radius bins to evaluate

    Returns array of correlation data at each radius"""

    directors = data[:, 2, :] - data[:, 0, :]
    theta_array, phi_array = find_angles(directors)

    max_separation = np.linalg.norm(box_dim) / 2

    bin_width = max_separation / bin_num
    separation_bins = np.linspace(0, max_separation, bin_num, endpoint=False)
    correlation_data = np.zeros_like(separation_bins)

    for n, radius in enumerate(separation_bins):
        order_param_sum = 0
        sample_size = 0

        for i in range(N_molecules):
            running_tot = 0

            separation_array = find_separations(data[:, 2, :], data[i, 2, :], box_dim)
            relevant_theta = np.ma.masked_where(
                np.logical_or(
                    (separation_array < radius),
                    (separation_array > (radius + bin_width)),
                ),
                theta_array,
            )
            relevant_phi = np.ma.masked_where(
                np.ma.getmask(relevant_theta), phi_array
            )  # applies the mask of theta on phi

            for m in range(-order, order + 1):  # from -l to l
                harmonic_at_i = sph_harm(m, order, theta_array[i], phi_array[i]).real
                harmonics_sum = spherical_harmonic_sum(
                    relevant_theta, relevant_phi, order, m
                )
                # remove i=j term from sum, then multiply by harmonic for molecule i
                running_tot += (harmonics_sum - harmonic_at_i) * harmonic_at_i

            order_param_sum += (4 * np.pi / (2 * order + 1)) * running_tot
            sample_size += relevant_theta.count()  # number of pairs sampled

        correlation_data[n] = order_param_sum / sample_size

        print("    radius = " + str(int(radius)) + "/" + str(int(max_separation)))
    print(correlation_data)
    return separation_bins, correlation_data


# READ MOLECULE POSITIONS

order_param_values = np.zeros(len(time_range))
volume_values = np.full(len(time_range), np.nan)  # new array of NaN
for i, time in enumerate(time_range):  # interate over dump files
    data_file = open(FILE_ROOT + str(time) + ".dump", "r")
    extract_atom_data = False  # start of file doesn't contain particle values
    extract_box_data = False  # start of file doesn't contain box dimension

    box_volume = 1
    box_dimensions = []  # to store side lengths of box for period boundary adjustment
    rod_positions = np.zeros((N_molecules, 3, 3))
    """Indices are Molecule Number; Atom number 1st/mid/last ; Positional coord index"""

    for line in data_file:
        if "ITEM: BOX" in line:  # to start reading volume data
            extract_box_data = True
            extract_atom_data = False
            continue  # don't attempt to read this line

        if "ITEM: ATOMS" in line:  # to start reading particle data
            extract_box_data = False
            extract_atom_data = True
            continue

        if extract_box_data and not extract_atom_data:
            # evaluate before particle values
            # each line gives box max and min in a single axis
            box_limits = []
            for d in line.split():  # separate by whitespace
                try:
                    box_limits.append(float(d))
                except ValueError:
                    pass  # any non-floats in this line are ignored
            box_volume *= box_limits[1] - box_limits[0]
            # multiply box volume by length of this dimension of box
            box_dimensions.append(box_limits[1] - box_limits[0])

        if extract_atom_data and not extract_box_data:
            # evaluate after box dimension collection
            # each line is in the form "id mol type x y z vx vy vz"
            particle_values = []
            for t in line.split():  # separate by whitespace
                try:
                    particle_values.append(float(t))
                except ValueError:
                    pass  # any non-floats in this line are ignored

            # # Save positional coordatinates of end particles - REGULAR
            # if int(particle_values[2]) == 1:  # first particle
            #     rod_positions[int(particle_values[1]) - 1, 0, :] = particle_values[3:6]
            # if int(particle_values[2]) == int((mol_length + 1) / 2):  # central particle
            #     rod_positions[int(particle_values[1]) - 1, 1, :] = particle_values[3:6]
            # if int(particle_values[2]) == mol_length:  # last particle
            #     rod_positions[int(particle_values[1]) - 1, 2, :] = particle_values[3:6]

            # Save positional coordatinates of end particles - CLOSE
            centre = (mol_length + 1) / 2
            if int(particle_values[2]) == int(centre - 1):
                rod_positions[int(particle_values[1]) - 1, 0, :] = particle_values[3:6]
            if int(particle_values[2]) == int(centre):  # central particle
                rod_positions[int(particle_values[1]) - 1, 1, :] = particle_values[3:6]
            if int(particle_values[2]) == int(centre + 1):
                rod_positions[int(particle_values[1]) - 1, 2, :] = particle_values[3:6]

    data_file.close()  # close data_file for time step t
    volume_values[i] = box_volume
    separation_bins, correlation_data = correlation_func(
        rod_positions, box_dimensions, SEPARATION_BIN_NUM, order=2
    )  # evaluate order param at time t

    tot_plot_num = len(time_range)
    colors = plt.cm.cividis(np.linspace(0, 1, tot_plot_num))
    if i == 0:
        continue  # don't plot this case
    plt.plot(
        separation_bins, correlation_data, color=colors[i],
    )

    print("T = " + str(time) + "/" + str(run_time))

sm = plt.cm.ScalarMappable(cmap=cm.cividis, norm=plt.Normalize(vmin=0, vmax=run_time))
cbar = plt.colorbar(sm)
cbar.ax.set_ylabel("Number of Time Steps", rotation=270, labelpad=15)

plt.title("Pairwise Angular Correlation Function")
plt.xlabel("Particle Separation")
plt.ylabel("Correlation Function")
plt.savefig("correlation_func_FT.png")
plt.show()
