"""Fourier transformmethod to calculate the pair-wise orientational order correlation function over the 
particle separation (y component only here). As detailed by Daan Frenkel, uses spherical harmonics so 
avoid costly angle calculations (which are O(N^2)) as well as fourier transform methods. 
    Uses the end-to-end molecule vector as the director, and also includes channels approach.
There may be normalisation issues with this, as p(m) is not normalised, but this is not relevant for our work"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from scipy.ndimage import uniform_filter1d  # for rolling average
from scipy.special import sph_harm
from phase_plot import vol_frac

FILE_ROOT = "output_T_0.5_time_"  # two underscores to match typo in previous code
SAMPLING_FREQ = 200  # only samples one in X files (must be integer)

POSITION_BIN_NUM = 4  # number of bins for position dependance pair-wise correlation
# For fast fourier transform (not implemented here), this is optimised if a power of 2
CHANNEL_NUM = 4  # N number of channels in each direction of the x-z plane (N^2 total)
# this should be order O(N^2/3), while POSITION_BIN_NUM should be 10 to 100 * O(N^1/3)

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


def find_angles(vec_array):
    """Finds spherical angles of cartesian position. Returns theta and phi in N x 2 array
    
    vec_array should be of size N x 3, for N angles (ie N particles)
    Choice of (normalised) base vector is arbitrary in order param formulation"""

    assert vec_array.shape[1] == 3, "Vectors should be three dimensional"
    radius = np.linalg.norm(vec_array, axis=1)
    theta = np.arccos(vec_array[:, 2] / radius)
    phi = np.arctan2(vec_array[:, 1], vec_array[:, 0])
    return theta, phi


def rms_fourier_transform(
    m_vector_array, column_indices, sph_harm_array, k_vector, cell_num, channel_num
):
    """Computes the rms of the discrete FT of angle density, averaging over N^2 channels

    Input Nx1 arrays of m_vectors, the channel indices, and the spherical harmonics, per molecule for N molecules
    Also input - the wavevector to calculate the FT at.
               - the number of channels (to average over)
               - number of cells (ie fourier components)"""

    ft_result = np.zeros(
        (channel_num, channel_num), dtype=np.complex
    )  # Use for running total

    for i in range(len(sph_harm_array)):  # sum over all molecules in sample
        # Add to running total in that channel
        ft_result[column_indices[0, i], column_indices[1, i]] += sph_harm_array[
            i
        ] * np.exp(
            2j
            * np.pi
            * np.dot(k_vector, m_vector_array[i, :])  # / cell_num  # CHECK THIS
        )
        # replace m_vector_array[i, :] with [0, m_vector_array[i, 1], 0] for 1D version (y component only)

    rms_result = np.mean(ft_result * np.conj(ft_result))
    return rms_result


def compute_spherical_harmonics(
    data,
    box_dim,
    cell_num,
    order=0,
    sub_order=0,
    detect_errors=True,
    print_warning=False,
    m_vectors_only=False,
):
    """Computes spherical harmonic of molecule orientation, and m vector, for each molecule
    
    Input data will be rod_positions array of size Molecule Number x 3 x 3
    First index gives molecule number (up to size N_molecules)
    Second index gives particle number within molecule (first/centre/last)
    Third index gives the component of the position (x,y,z)

    Also input  'box_dim' - list of simulation box dimensions
                'cell_num' - number of cells for discrete FT
                'order' - order of correlation function to compute (l)
                'suborder' - degree of harmonic (typically m, where m = -l, ..., l)
                'detect_erros' - boolean whether to reflect spurious position data back into cell
                'print_warning' - boolean whether to print warning about spurious positions
                'm_vectors_only' - Allows computation of m_vectors only - used if channel_num != cell_num

    Returns two Nx1 arrays of m vectors and spherical harmonics respectively for each of N molecules"""

    directors = data[:, 2, :] - data[:, 0, :]
    theta_array, phi_array = find_angles(directors)
    sph_harm_array = np.full_like(directors[:, 0], np.nan)  # Nx1 array
    m_vector_array = np.full_like(directors, np.nan)  # Nx3 array

    position_values = data[:, 1, :] + box_dim / 2
    # change origin of cell to corner so no negative values
    cell_dim = box_dim / cell_num

    # GENERATE NUMBER DENSITY ARRAY
    for n, pos in enumerate(position_values):
        m_vector_array[n, :] = (
            pos // cell_dim
        )  # integer steps from corner (origin) of region
        if not m_vectors_only:
            sph_harm_array[n] = sph_harm(
                sub_order, order, theta_array[n], phi_array[n]
            ).real

        if detect_errors:
            for j in range(3):
                if box_dim[j] < pos[j]:  # Particle outside box - floating point error?
                    if print_warning:
                        print(
                            f"  Warning: Particle detected {(pos[j] - box_dim[j]):.{4}}"
                            + " outside box (possible floating point error) - reflected in boundary"
                        )

                    pos[j] = (2 * box_dim[j]) - pos[j]  # reflect inside box
                    m_vector_array[n, :] = pos // cell_dim  # redo previous calculations
    # print(m_vector_array, sph_harm_array)
    if m_vectors_only:
        return m_vector_array
    else:
        return m_vector_array, sph_harm_array


def correlation_func(pos_data, box_dim, cell_num, channel_num, delta_m_list, order):
    """Input position data in array of size Molecule Number x 3 x 3

    Input data will be rod_positions array which stores input data   
    First index gives molecule number (up to size N_molecules)
    Second index gives particle number within molecule (first/last)
    Third index gives the component of the position (x,y,z)

    Also input  'order' - order of correlation function to compute
                'box_dim' - list of simulation box dimensions
                'cell_num' - number of cells for discrete FT
                'delta_m_list' - list of delta_m values to evaluate function at

    Returns list of angle correlation func values at each delta_m"""

    correlation_data = np.zeros_like(delta_m_list, dtype=np.complex)

    for i, delta_m in enumerate(delta_m_list):
        delta_m_vector = np.array([0, delta_m, 0])

        outer_sum_tot = 0
        for sub_order in range(-order, order + 1):  # m = -l, -l+1, ..., l-1, l
            if i == 0 and sub_order == -order:  # Only prints warning on first occurance
                print_warning_bool = True
            else:
                print_warning_bool = False

            m_vector_array, sph_harm_array = compute_spherical_harmonics(
                pos_data,
                box_dim,
                cell_num,
                order,
                sub_order,
                detect_errors=True,
                print_warning=print_warning_bool,
                m_vectors_only=False,
            )

            if cell_num == channel_num:
                # Can use m_vector indices to allocate particles in channels
                column_indices = np.array([m_vector_array[:, 0], m_vector_array[:, 2]])
            else:
                # Recompute the m_vectors to find channe; indices for each particle
                channel_indices = compute_spherical_harmonics(
                    pos_data,
                    box_dim,
                    channel_num,
                    order,
                    sub_order,
                    detect_errors=True,
                    print_warning=print_warning_bool,
                    m_vectors_only=True,
                )
                column_indices = np.array(
                    [channel_indices[:, 0], channel_indices[:, 2]]
                )
            column_indices = column_indices.astype(int)  # required for use in indexing

            for delta_m_value in delta_m_list:
                # sum over k wavevectors, given by inverse of delta_m values
                inner_sum_tot = 0

                centred_m_value = delta_m_value + 1 / 2
                # this gives vector to centre of cell, not corner, and avoids /0 in next line
                k_value = (2 * np.pi / cell_num) * np.reciprocal(centred_m_value)
                k_vector = np.array([0, k_value, 0])  # again aligned along y axes

                rms_ft_density = rms_fourier_transform(
                    m_vector_array,
                    column_indices,
                    sph_harm_array,
                    k_vector,
                    cell_num,
                    channel_num,
                )

                inner_sum_tot += rms_ft_density * np.exp(
                    -2j * np.pi * np.dot(k_vector, delta_m_vector) / cell_num
                )
                # print(ft_density)

            outer_sum_tot += (inner_sum_tot) / (cell_num)  # only power of -1 as 1D FT

        correlation_data[i] = ((4 * np.pi) / (2 * order + 1)) * outer_sum_tot

    return np.real(correlation_data)


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

    # delta_m_list = np.linspace(0, box_dimensions[1], POSITION_BIN_NUM, endpoint=False)
    delta_m_list = np.linspace(0, POSITION_BIN_NUM, endpoint=False)
    # USE DISPLACEMENTS ALOG Y AXIS ONLY FOR THIS
    y_step = box_dimensions[1] / POSITION_BIN_NUM

    correlation_data = correlation_func(
        rod_positions,
        np.array(box_dimensions),
        POSITION_BIN_NUM,
        CHANNEL_NUM,
        delta_m_list,
        order=2,  # Second legendre polynomial for nematic order
    )

    tot_plot_num = len(time_range)
    colors = plt.cm.cividis(np.linspace(0, 1, tot_plot_num))
    # if i == 0:
    #     continue  # don't plot this case
    plt.plot(
        delta_m_list * y_step, correlation_data, color=colors[i],
    )

    print("T = " + str(time) + "/" + str(run_time))

sm = plt.cm.ScalarMappable(cmap=cm.cividis, norm=plt.Normalize(vmin=0, vmax=run_time))
cbar = plt.colorbar(sm)
cbar.ax.set_ylabel("Number of Time Steps", rotation=270, labelpad=15)

plt.title("Pairwise Angular Correlation Function")
plt.xlabel("Particle Separation")
plt.ylabel("Correlation Function")
plt.savefig("correlation_func_FT_channels_short2.png")
plt.show()
