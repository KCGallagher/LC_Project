"""Fourier transformmethod to calculate the pair-wise orientational order correlation function over the 
particle separation (y component only here). As detailed by Daan Frenkel, uses spherical harmonics so 
avoid costly angle calculations (which are O(N^2)) as well as fast fourier transform methods. 
    Uses the end-to-end molecule vector as the director, but not including channels approach.
There may be normalisation issues with this, as p(m) is not normalised, but this is not relevant for our work"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from scipy.ndimage import uniform_filter1d  # for rolling average
from scipy.special import sph_harm
from scipy.fft import fft
from phase_plot import vol_frac

import time

start_time = time.time()

FILE_ROOT = "output_T_0.5_time_"  # two underscores to match typo in previous code
SAMPLING_FREQ = 200  # only samples one in X files (must be integer)

POSITION_BIN_NUM = 64  # number of bins for position dependance pair-wise correlation
# For fast fourier transform (not implemented here), this is optimised if a power of 2

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


def rms_fourier_transform(sph_harm_array):
    """Computes the rms of the discrete FT of angle density
    Input Nx1 array of spherical harmonics, per molecule for N molecules"""

    ft_result = fft(sph_harm_array)

    rms_result = np.mean(ft_result * np.conj(ft_result))
    return rms_result


def compute_spherical_harmonics(theta_array, phi_array, order, sub_order):
    """Computes spherical harmonic of molecule orientation, for each molecule
    
    Input data will be two angle arrays of size Molecule Number x 1
    First array gives theta values, second array gives phi values

    Also input  'order' - order of correlation function to compute (l)
                'suborder' - degree of harmonic (typically m, where m = -l, ..., l)

    Returns Nx1 array of spherical harmonics for each of N molecules"""

    mol_number = len(theta_array)
    sph_harm_array = np.full((mol_number, 1), np.nan)  # Nx1 array

    for n in range(mol_number):
        sph_harm_array[n] = sph_harm(
            sub_order, order, theta_array[n], phi_array[n]
        ).real

    return sph_harm_array


def correlation_func(pos_data, box_dim, cell_num, delta_m_list, order):
    """Input position data in array of size Molecule Number x 3 x 3

    Input data will be rod_positions array which stores input data   
    First index gives molecule number (up to size N_molecules)
    Second index gives particle number within molecule (first/last)
    Third index gives the component of the position (x,y,z)

    Also input  'box_dim' - list of simulation box dimensions
                'cell_num' - number of cells for discrete FT
                'delta_m_list' - list of delta_m values to evaluate function at
                'order' - order of correlation function to compute

    Returns list of angle correlation func values at each delta_m"""

    correlation_data = np.zeros_like(delta_m_list, dtype=np.complex)

    directors = pos_data[:, 2, :] - pos_data[:, 0, :]
    theta_array, phi_array = find_angles(directors)

    # Calculate spherical harmonics, m vectors and channel indices immediately for all suborders to save time later
    sph_harm_array = np.full(
        (len(theta_array), (2 * order + 1)), np.nan
    )  # Nx5 array for order = 2
    for sub_order in range(-order, order + 1):  # m = -l, -l+1, ..., l-1, l
        sub_order_index = sub_order + order  # from 0 to 2l inclusive
        sph_harm_array[:, sub_order_index] = np.squeeze(
            compute_spherical_harmonics(theta_array, phi_array, order, sub_order,)
        )

    for i, delta_m in enumerate(delta_m_list):
        delta_m_vector = np.array([0, delta_m, 0])
        outer_sum_tot = 0

        for delta_m_value in delta_m_list:
            # sum over k wavevectors, given by inverse of delta_m values
            inner_sum_tot = 0

            centred_m_value = delta_m_value + 1 / 2
            # this gives vector to centre of cell, not corner, and avoids /0 in next line
            k_value = (2 * np.pi / cell_num) * np.reciprocal(centred_m_value)
            k_vector = np.array([0, k_value, 0])  # again aligned along y axes

            for sub_order in range(-order, order + 1):  # m = -l, -l+1, ..., l-1, l

                rms_ft_density = rms_fourier_transform(
                    sph_harm_array[:, sub_order + order]
                )
                inner_sum_tot += rms_ft_density

            outer_sum_tot += (
                ((4 * np.pi) / (2 * order + 1))
                * (inner_sum_tot)
                * np.exp(-2j * np.pi * np.dot(k_vector, delta_m_vector) / cell_num)
            )

        correlation_data[i] = outer_sum_tot  # / (cell_num)  # only power of -1 as 1D FT
    return np.real(correlation_data)


# READ MOLECULE POSITIONS

order_param_values = np.zeros(len(time_range))
volume_values = np.full(len(time_range), np.nan)  # new array of NaN
max_value = 0  # for comparison of scaling between graphs
for i, file_time in enumerate(time_range):  # interate over dump files
    data_file = open(FILE_ROOT + str(file_time) + ".dump", "r")
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
        delta_m_list,
        order=2,  # Second legendre polynomial for nematic order
    )

    tot_plot_num = len(time_range)
    colors = plt.cm.cividis(np.linspace(0, 1, tot_plot_num))
    if i == 0:
        continue  # don't plot this case
    plt.plot(
        delta_m_list * y_step, correlation_data, color=colors[i],
    )
    if max(correlation_data) > max_value:
        max_value = max(correlation_data)
    print("T = " + str(file_time) + "/" + str(run_time))

end_time = time.time()
print("Total time: " + str(end_time - start_time) + " seconds")

sm = plt.cm.ScalarMappable(cmap=cm.cividis, norm=plt.Normalize(vmin=0, vmax=run_time))
cbar = plt.colorbar(sm)
cbar.ax.set_ylabel("Number of Time Steps", rotation=270, labelpad=15)

plt.title("Max_Value = " + str(round(max_value, 2)))
plt.xlabel("Particle Separation")
plt.ylabel("Correlation Function")
plt.savefig("test_image_fft_M" + str(POSITION_BIN_NUM) + "_N1c.png")
plt.show()
