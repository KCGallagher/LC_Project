"""Calculates the diffusion matrix over each equillibration run. 
Accounts for additional displacement when crossing the periodic boundary conditions

This file is adapted for no contraction periods, and biaxial phase structure"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d  # for rolling average
from scipy.stats import linregress  # for linear regression
from phase_plot import vol_frac

FILE_ROOT = "output_T_0.5_time_"  # two underscores to match typo in previous code
PLOT_BEST_FIT = True

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
        run_num_tot = 0  # counts number of runs
        for t in line.split():
            try:
                mix_steps_values.append(int(t))
                run_num_tot += 1
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
run_time = tot_mix_time + run_num_tot * equilibrium_time
time_range = np.arange(0, int(run_time), int(dump_interval))
eq_range = np.arange(0, int(equilibrium_time + dump_interval), int(dump_interval))
print(
    "N_molecules, run_time, dump_interval = "
    + str((N_molecules, run_time, dump_interval))
)

# GENERATE LIST OF TIME STEPS TO SAMPLE

sampling_times = np.zeros(len(mix_steps_values) + 1)
# Gives start and end times for each equillibrium run
time_counter = 0
for i in range(len(mix_steps_values)):
    time_counter += equilibrium_time
    sampling_times[i + 1] = time_counter
print("Sampling Times: " + str(sampling_times))

assert time_counter == run_time, "Unexpected result in sampling times"

# CALCULATE THE RMS DISPLACEMENT
def periodic_bc_displacement(current_pos, previous_pos, box_dim, tolerance=0.5):
    """Input position data in arrays of size Molecule Number x 3, and list of box_dim

    Input data will be com_positions array which stores input data   
    First index gives molecule number
    Second index gives the component of the position (x,y,z)
    'tolerance' is the maximum allowed step size (without assuming boundary crossing)
        It is measured as a fraction of the box dimensions, default is 0.5

    Returns an additional displacement vector which track displacement over boundaries of box"""
    delta_x = current_pos - previous_pos
    output_displacement = np.zeros_like(delta_x)
    for i in range(len(box_dim)):
        # adds displacement equal to box dimension in direction of boundary crossing
        output_displacement[:, i] += np.where(
            delta_x[:, i] > tolerance * box_dim[i], -box_dim[i], 0
        )  # crossings in negative axis direction (to give large +ve delta_x)
        output_displacement[:, i] += np.where(
            delta_x[:, i] < -1 * tolerance * box_dim[i], box_dim[i], 0
        )  # crossings in positive axis direction
    return output_displacement


def rms_displacement(pos_t, pos_0):
    """Input data in array of size Molecule Number x 3, and list of box_dim

    Input data will be com_positions array which stores input data   
    First index gives molecule number
    Second index gives the component of the position (x,y,z)

    Returns average displacement in each coordinate axis"""

    rms_vector = np.abs((pos_t - pos_0))
    return np.mean(rms_vector, axis=0)


# READ MOLECULE POSITIONS

axis_labels = ["x", "y", "z"]

volume_values = np.full(len(time_range), np.nan)  # new array of NaN

# for ongoing measurements:
rms_disp_values = np.full((run_num_tot, len(eq_range), 3, 3), np.nan)
time_origin = 0
run_num = 0

# for sampled measurements:
sampled_D_values = np.full((len(mix_steps_values), 3, 3), np.nan)
sampled_vol_values = np.full(len(mix_steps_values), np.nan)
equilibrium_flag = False  # denotes whether system is currently in an equillibrium run

extra_displacement = np.zeros((N_molecules, 3))
# additional values to account for crossing the boundaries of the box

director_vectors = np.full((len(mix_steps_values) - 1, 3), np.nan)
bisector_vectors = np.full((len(mix_steps_values) - 1, 3), np.nan)
normal_vectors = np.full((len(mix_steps_values) - 1, 3), np.nan)

for i, time in enumerate(time_range):  # interate over dump files
    data_file = open(FILE_ROOT + str(time) + ".dump", "r")
    extract_atom_data = False  # start of file doesn't contain particle values
    extract_box_data = False  # start of file doesn't contain box dimension

    box_volume = 1
    box_dimensions = []  # to store side lengths of box for period boundary adjustment
    rod_positions = np.zeros((N_molecules, 3, 3))
    """Indices are Molecule Number;  Positional coord index"""

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

    com_positions = rod_positions[:, 1, :]
    data_file.close()  # close data_file for time step t
    volume_values[i] = box_volume

    if time != 0:  # GIVE PREVIOUS DISPLACEMENT
        extra_displacement += periodic_bc_displacement(
            com_positions, previous_positions, box_dimensions
        )

    if time in sampling_times:  # MEASURE SAMPLING POINTS
        print("T = " + str(time) + "/" + str(run_time))
        sample_index = np.where(sampling_times == time)

        if time != 0:  # RUN ENDING ROUTINE FIRST
            # print(extra_displacement) # useful to check you aren't getting silly values/multiple crossings
            sampled_rms = rms_displacement(
                com_positions + extra_displacement, initial_sample,
            )  # initial sample taken from previous iteration in if clause
            sampled_D_values[sample_index, :] = sampled_rms / (6 * equilibrium_time)
            # D value for i-th equillibration period
            run_num += 1

        if time != run_time:  # RUN STARTING ROUTINE FOR SAMPLING
            initial_sample = com_positions
            sampled_vol_values[sample_index] = box_volume

            extra_displacement = np.zeros_like(com_positions)  # reset for each sample

            run_origin = i  # gives time index for the start of each run

    # MEASURE ONGOING RMS DISPLACEMENT
    position_vector = rms_displacement(
        com_positions + extra_displacement,  # takes current values
        initial_sample,  # reset for each eq run
    )
    rms_disp_values[run_num, i - run_origin, :, :] = np.outer(
        position_vector, position_vector
    )

    previous_positions = com_positions


# GENERATE DIFFUSION PLOTS
def label_maker(label, plot_index):
    """Returns label if plot_index == 0, null otherwise"""
    if plot_index == 0:  # for legend
        return label
    else:
        return None


plot_list = range(1, run_num_tot, 1)  # runs to plot (inc step if too many runs)
sampled_vol_frac = vol_frac(sampled_vol_values, mol_length, N_molecules)

fig, axs = plt.subplots(nrows=1, ncols=len(plot_list), sharey=True, figsize=(10, 5))
for plot_index, data_index in enumerate(plot_list):
    axs[plot_index].set_title(
        r"$\phi =$" + "{:.2f}".format(sampled_vol_frac[data_index])
    )

    eq_time_values = np.array(eq_range)
    eq_time_values[0] = eq_time_values[1]  # remove zero so log can be evaluated

    plot_data = np.zeros_like(rms_disp_values[data_index, :, :, 0])
    # only need a 3x1 vector for each time point in each sample, not a 3x3 matrix

    for i in range(len(eq_range)):
        for j in range(3):
            plot_data[i, j] = rms_disp_values[
                data_index, i, j, j
            ]  # pick out diagonal terms

    # rms_disp has values for all timesteps in sample. so apply the same dot operation to all vectors
    for j in range(3):
        axs[plot_index].loglog(
            eq_range, plot_data[:, j], label=label_maker(axis_labels[j], plot_index)
        )

        if PLOT_BEST_FIT:
            slope, intercept, r_value, p_value, std_err = linregress(
                np.log10(eq_time_values[1:-1]), np.log10(plot_data[1:-1, j])
            )  # remove first and last values in array as nan at end and zero at start
            label = str(axis_labels[j]) + " (Fit)"
            axs[plot_index].plot(
                eq_time_values,
                (eq_time_values ** slope) * (10 ** intercept),
                label=label_maker(label, plot_index),
                linestyle="dashed",
            )

            print(
                "For vol frac = "
                + "{:.2f}".format(sampled_vol_frac[data_index])
                + ", slope = "
                + "{:.2f}".format(slope)
                + " for axis "
                + str(axis_labels[j])
            )  # can add this onto graph with plt.annotate if desired

axs[int(len(plot_list) / 2)].set_xlabel("Time Step")  # use median of plot_list
axs[0].set_ylabel(r"RMS Displacement ($\langle x_{i}\rangle^{2}$)")
fig.legend(loc="center right")
plt.savefig("rms_displacement_runwise_bf_cf.png")
plt.show()

# print("Mean Director: " + str(np.mean(director_vectors, axis=0)))
# print("Mean Bisector: " + str(np.mean(bisector_vectors, axis=0)))
# print(
#     "Mean Normal : " + str(np.mean(normal_vectors, axis=0))
# )  # NaN if USE_SYS_BASIS = False

