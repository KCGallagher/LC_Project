"""Fourier transform method to calculate the pair-wise orientational order correlation function over the 
particle separation (y component only here). As detailed by Daan Frenkel, uses spherical harmonics so 
avoid costly angle calculations (which are O(N^2)) as well as fast fourier transform methods. 
    Uses the end-to-end molecule vector as the director, and including channels approach (for channels along the y axis)
There may be normalisation issues with this, as p(m) is not normalised, but this is not relevant for our work"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from scipy.ndimage import uniform_filter1d  # for rolling average
from scipy.special import sph_harm
from scipy.fft import fft, fftfreq, fftshift, ifft
from phase_plot import vol_frac

import time

start_time = time.time()

FILE_ROOT = "output_T_0.5_time_"  # two underscores to match typo in previous code
SAMPLING_FREQ = 100  # only samples one in X files (must be integer)

POSITION_BIN_NUM = 512  # (M) bin number for position dependance pair-wise correlation
# For fast fourier transform (not implemented here), this is optimised if a power of 2
CHANNEL_NUM = (
    128  # (H) number of channels in each direction of the x-z plane (H^2 total)
)
# this should be order O(N^2/3) = 128, while POSITION_BIN_NUM should be 10 to 100 * O(N^1/3) = 512

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
    """Finds spherical angles of cartesian position. Returns theta and phi in two N x 1 arrays
    
    vec_array should be of size N x 3, for N angles (ie N particles)
    Choice of (normalised) base vector is arbitrary in order param formulation"""

    assert vec_array.shape[1] == 3, "Vectors should be three dimensional"
    radius = np.linalg.norm(vec_array, axis=1)
    theta = np.arccos(vec_array[:, 2] / radius)
    phi = np.arctan2(vec_array[:, 1], vec_array[:, 0])
    return theta, phi


def rms_fourier_transform(sph_harm_array):
    """Computes the rms of the discrete FT of angle density
    Input HxMxH array of (summed) spherical harmonics, for H channels and M cells
    Returns Mx1 array of fourier transform values at each of M frequencies, averaged across channels"""

    input_array_shape = sph_harm_array.shape
    summed_fourier_components = np.zeros(
        input_array_shape[1], dtype=np.complex
    )  # running total for mean

    for channel_index in np.ndindex((input_array_shape[0], input_array_shape[2])):
        ft_channel = fft(
            sph_harm_array[channel_index[0], :, channel_index[1]]
        )  # FT of all cells on y axis for given channel
        ms_channel = np.multiply(
            ft_channel, np.conj(ft_channel)
        )  # find magnitude squared
        summed_fourier_components += ms_channel

    mean_fourier_components = summed_fourier_components / (
        input_array_shape[0] * input_array_shape[2]
    )
    return summed_fourier_components  # no need for mean?


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


def correlation_func(pos_data, box_dim, cell_num, channel_num, delta_m_list, order):
    """Input position data in array of size Molecule Number x 3 x 3

    Input data will be rod_positions array which stores input data   
    First index gives molecule number (up to size N_molecules)
    Second index gives particle number within molecule (first/last)
    Third index gives the component of the position (x,y,z)

    Also input  'box_dim' - list of simulation box dimensions
                'cell_num' - number of cells for discrete FT
                'delta_m_list' - list of delta_m values to evaluate function at
                'order' - order of correlation function to compute

    Returns list of angle correlation func values at each delta_m, as well as corresponding distances"""

    correlation_data = np.zeros_like(delta_m_list, dtype=np.complex)

    directors = pos_data[:, 2, :] - pos_data[:, 0, :]
    theta_array, phi_array = find_angles(directors)

    cell_dim = box_dim / np.array([channel_num, cell_num, channel_num])

    # Calculate spherical harmonics, m vectors and channel indices immediately for all suborders to save time later
    sph_harm_array = np.full(
        (len(theta_array), (2 * order + 1)), np.nan
    )  # Nx5 array for order = 2

    for sub_order in range(-order, order + 1):  # m = -l, -l+1, ..., l-1, l
        sub_order_index = sub_order + order  # from 0 to 2l inclusive
        sph_harm_array[:, sub_order_index] = np.squeeze(
            compute_spherical_harmonics(theta_array, phi_array, order, sub_order,)
        )

    # Form density matrix, calculate the sum of all spherical harmonics in a given cell, at each suborder
    sph_density_array = np.zeros((channel_num, cell_num, channel_num, (2 * order + 1)))
    # H*M*H array for each of (2l + 1) suborders - watch size of this array

    for index, pos in enumerate(pos_data[:, 1, :]):  # iterate over CoM positions
        m_vector = pos // cell_dim  # integer steps from corner (origin) of region
        m_vector = m_vector.astype(int)
        # This also acts as index for relevant cell in p(m)
        try:
            # sph_density_array[
            #     m_vector[0], m_vector[1], m_vector[2], :
            # ] += sph_harm_array[index, :] # explicit version of subsequent line
            sph_density_array[tuple(m_vector.astype(int))] += sph_harm_array[index, :]
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
                    m_vector = m_vector.astype(int)

                    sph_density_array[tuple(m_vector.astype(int))] += sph_harm_array[
                        index, :
                    ]
            if not problem_identified:
                raise  # I.e. error was not due to this suspected floating point issue

    # Compute FT of the sph_density_array, RMS averaged over all channels
    rms_ft_density_array = np.full(
        (cell_num, (2 * order + 1)), np.nan, dtype=np.complex
    )
    for sub_order in range(-order, order + 1):  # m = -l, -l+1, ..., l-1, l
        sub_order_index = sub_order + order  # from 0 to 2l inclusive
        rms_ft_density_array[:, sub_order_index] = rms_fourier_transform(
            sph_density_array[:, :, :, sub_order_index]
        )
    rms_ft_density_sums = np.sum(rms_ft_density_array, axis=1)  # sum over suborders

    freq_components = fftfreq(
        rms_ft_density_array[:, sub_order_index].size, cell_dim[1]
    )
    end_index = len(freq_components)
    pos_freq_components = freq_components[
        1 : int((end_index + 1) / 2)
    ]  # positive, non-zero freq components only

    for i, delta_m in enumerate(delta_m_list):
        outer_sum_tot = 0

        for f_index, freq in enumerate(pos_freq_components):
            # sum over k wavevectors (defined along y axis), given by inverse of delta_m values

            rms_ft_density = rms_ft_density_sums[f_index]

            outer_sum_tot += (
                ((4 * np.pi) / (2 * order + 1))
                * (rms_ft_density)
                * np.exp(-2j * np.pi * freq * delta_m / cell_num)
            )

        correlation_data[i] = outer_sum_tot / len(
            pos_freq_components
        )  # i.e. divide by total number of freq comp.

    # USE DISPLACEMENTS ALOG Y AXIS ONLY FOR THIS
    distance_list = delta_m_list * cell_dim[1]

    if False:  # for automatic ifft
        distance_list = np.linspace(
            0, box_dimensions[1], cell_num, endpoint=False
        )  # plotting testing
        correlation_data = ((4 * np.pi) / (2 * order + 1)) * ifft(rms_ft_density_sums)
    return (
        distance_list,
        np.real(correlation_data) / len(theta_array),
    )  # normalise by N (N_molecules)


# READ MOLECULE POSITIONS

order_param_values = np.zeros(len(time_range))
volume_values = np.full(len(time_range), np.nan)  # new array of NaN
max_value = 0  # of correlation func, for comparison of scaling between graphs
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

    delta_m_list = np.arange(0, POSITION_BIN_NUM)  # increase in integer steps

    plotting_distance_list, correlation_data = correlation_func(
        rod_positions,
        np.array(box_dimensions),
        POSITION_BIN_NUM,
        CHANNEL_NUM,
        delta_m_list,
        order=2,  # Second legendre polynomial for nematic order
    )

    tot_plot_num = len(time_range)
    colors = plt.cm.cividis(np.linspace(0, 1, tot_plot_num))
    if i == 0:
        continue  # don't plot this case
    plt.plot(
        plotting_distance_list, correlation_data, color=colors[i],
    )
    if max(correlation_data) > max_value:
        max_value = max(correlation_data)
    print("T = " + str(file_time) + "/" + str(run_time))

end_time = time.time()
print("Total time: " + str(round(end_time - start_time, 2)) + " seconds")

sm = plt.cm.ScalarMappable(cmap=cm.cividis, norm=plt.Normalize(vmin=0, vmax=run_time))
cbar = plt.colorbar(sm)
cbar.ax.set_ylabel("Number of Time Steps", rotation=270, labelpad=15)

plt.title("Max_Value = " + str(round(max_value, 2)))
plt.xlabel("Particle Separation")
plt.ylabel("Correlation Function")
plt.savefig(
    "test_image_fft2_M" + str(POSITION_BIN_NUM) + "_H" + str(CHANNEL_NUM) + "norm.png"
)
plt.show()
