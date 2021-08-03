"""Fourier transform method to calculate the autocorrelation of number density over the particle separation (y component only here)
Primarily used as a test case for the correlation plot fourier transform methods, but also relevant to smectic phase
There may be normalisation issues with this, as p(m) is not normalised, but this is not relevant for our work"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from scipy.ndimage import uniform_filter1d  # for rolling average
from phase_plot import vol_frac

FILE_ROOT = "output_T_0.5_time_"  # two underscores to match typo in previous code
SAMPLING_FREQ = 20  # only samples one in X files (must be integer) (20)
POSITION_BIN_NUM = 16  # number of bins for pair-wise correlation (16)

# For fourier transform, this is optimised if a power of 2

MANUAL_FT = False

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


def fourier_transform(data, k_vector, cell_num):
    """Computes the FT of a 1D array
    Manual implementation for testing - fft algorithm much faster"""
    ft_data = np.zeros_like(data, dtype=np.complex)
    assert len(data) == cell_num, "Expected M points in data array"

    for index, density in np.ndenumerate(data):
        ft_data[index] += density * np.exp(
            2j * np.pi * np.dot(k_vector, index) / cell_num
        )
        # need to sum over all vectors m, not scalars (k_value should also be a vector - this is a 3D FT)
    return ft_data


def autocorrelation_func(pos_data, box_dim, cell_num, delta_m_list):
    """Input data in array of size Molecule Number x 2, a list of box_dim,
    number of cells for discrete FT, and list of delta_m values to evaluate function at

    Input data will be rod_positions array which stores input data   
    Second index gives the component of the position (x,y,z)

    Returns value of autocorrelation func at each delta_m"""

    position_values = pos_data + box_dim / 2
    # change origin of cell to corner so no negative values
    cell_dim = box_dim / POSITION_BIN_NUM

    density_data = np.zeros((cell_num, cell_num, cell_num))

    # GENERATE NUMBER DENSITY ARRAY
    for pos in position_values:
        m_vector = pos // cell_dim  # integer steps from corner (origin) of region
        # This also acts as index for relevant cell in p(m)
        # print(pos, box_dim)
        try:
            density_data[tuple(m_vector.astype(int))] += 1
        except IndexError:
            problem_identified = False
            for i in range(3):
                if box_dim[i] < pos[i]:  # Particle outside box - floating point error?
                    problem_identified = True
                    print(
                        f"  Warning: Particle detected {(pos[i] - box_dim[i]):.{4}}"
                        + " outside box (possible floating point error) - reflected in boundary"
                    )

                    pos[i] = (2 * box_dim[i]) - pos[i]  # reflect inside box
                    m_vector = pos // cell_dim  # redo previous calculations
                    density_data[tuple(m_vector.astype(int))] += 1
            if not problem_identified:
                raise  # I.e. error was not due to this suspected floating point issue

    correlation_data = np.zeros_like(delta_m_list, dtype=np.complex)
    for i, delta_m in enumerate(delta_m_list):
        delta_m_vector = np.array([0, delta_m, 0])

        if not MANUAL_FT:  # use built in function
            ft_density = np.fft.fft(density_data)  # Replaces sum method

        for index, density in np.ndenumerate(density_data):
            centred_index = index + np.ones_like(index) / 2
            # this gives vector to centre of cell, not corner, and avoids /0 in next line
            k_vector = (2 * np.pi / cell_num) * np.reciprocal(centred_index)

            if MANUAL_FT:
                ft_density = fourier_transform(density_data, k_vector, cell_num)

            ave_density = np.mean(np.square(np.abs(ft_density)))
            correlation_data[i] += ave_density * np.exp(
                -2j * np.pi * np.dot(k_vector, delta_m_vector) / cell_num
            )

    return (
        np.real(correlation_data) / cell_num ** 3
    )  # IS IT CORRECT TO TAKE REAL PART HERE?


# READ MOLECULE POSITIONS

order_param_values = np.zeros(len(time_range))
volume_values = np.full(len(time_range), np.nan)  # new array of NaN
for i, time in enumerate(time_range):  # interate over dump files
    data_file = open(FILE_ROOT + str(time) + ".dump", "r")
    extract_atom_data = False  # start of file doesn't contain particle values
    extract_box_data = False  # start of file doesn't contain box dimension

    box_volume = 1
    box_dimensions = []  # to store side lengths of box for period boundary adjustment
    rod_positions = np.zeros((N_molecules, 3))
    """Indices are Molecule Number; Positional coord index"""

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

            # Save positional coordatinates of CoM of molecule:
            centre = (mol_length + 1) / 2
            if int(particle_values[2]) == int(centre):  # central particle
                rod_positions[int(particle_values[1]) - 1, :] = particle_values[3:6]

    data_file.close()  # close data_file for time step t
    volume_values[i] = box_volume

    # delta_m_list = np.linspace(0, box_dimensions[1], POSITION_BIN_NUM, endpoint=False)
    delta_m_list = np.linspace(0, POSITION_BIN_NUM, endpoint=False)
    # USE DISPLACEMENTS ALONG Y AXIS ONLY FOR THIS
    y_step = box_dimensions[1] / POSITION_BIN_NUM

    correlation_data = autocorrelation_func(
        rod_positions, np.array(box_dimensions), POSITION_BIN_NUM, delta_m_list
    )

    tot_plot_num = len(time_range)
    colors = plt.cm.cividis(np.linspace(0, 1, tot_plot_num))
    if i == 0:
        continue  # don't plot this case
    plt.plot(
        delta_m_list * y_step, correlation_data, color=colors[i],
    )

    print("T = " + str(time) + "/" + str(run_time))

sm = plt.cm.ScalarMappable(cmap=cm.cividis, norm=plt.Normalize(vmin=0, vmax=run_time))
cbar = plt.colorbar(sm)
cbar.ax.set_ylabel("Number of Time Steps", rotation=270, labelpad=15)

plt.title("Number Density over Contraction")
plt.xlabel("Particle Separation")
plt.ylabel("Number Density")
# plt.savefig("density_func_FT2.png")
plt.show()
