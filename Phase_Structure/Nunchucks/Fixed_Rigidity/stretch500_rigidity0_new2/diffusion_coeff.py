"""Calculates the diffusion coefficient over each equillibration run. 
Accounts for additional displacement when crossing the periodic boundary conditions"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d  # for rolling average
from scipy.stats import linregress  # for linear regression
from phase_plot import vol_frac

FILE_ROOT = "output_T_0.5_time_"  # two underscores to match typo in previous code
DIRECTIONAL_COEFF = True
REPEAT_CORRECTION = (
    False  # if you have repeated samples at the same volume fracion (no shrinking step)
)

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

sampling_times = np.zeros((len(mix_steps_values), 2))
# Gives start and end times for each equillibrium run
time_counter = 0
for i in range(len(mix_steps_values)):
    time_counter += mix_steps_values[i]
    sampling_times[i, 0] = time_counter
    time_counter += equilibrium_time
    sampling_times[i, 1] = time_counter

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


def rms_displacement(pos_t, pos_0, use_vector=False):
    """Input data in array of size Molecule Number x 3, and list of box_dim

    Input data will be com_positions array which stores input data   
    First index gives molecule number
    Second index gives the component of the position (x,y,z)

    If use_vector is false (default), returns rms displacement from initial displacement
    If use_vector is true, returns average displacement in each coordinate axis"""
    if use_vector:
        rms_vector = np.abs((pos_t - pos_0))
        return np.mean(rms_vector, axis=0)
    else:
        rms_value = np.linalg.norm((pos_t - pos_0))
        return np.mean(rms_value)


# READ MOLECULE POSITIONS

if DIRECTIONAL_COEFF:
    dimension_num = 3
    axis_labels = ["x", "y", "z"]
else:
    dimension_num = 1
    axis_labels = ["RMS"]

volume_values = np.full(len(time_range), np.nan)  # new array of NaN

# for ongoing measurements:
rms_disp_values = np.full((run_num_tot, len(eq_range), dimension_num), np.nan)
time_origin = 0
run_num = 0

# for sampled measurements:
sampled_D_values = np.full((len(mix_steps_values), dimension_num), np.nan)
sampled_vol_values = np.full(len(mix_steps_values), np.nan)
equilibrium_flag = False  # denotes whether system is currently in an equillibrium run

extra_displacement = np.zeros((N_molecules, 3))
# additional values to account for crossing the boundaries of the box

for i, time in enumerate(time_range):  # interate over dump files
    data_file = open(FILE_ROOT + str(time) + ".dump", "r")
    extract_atom_data = False  # start of file doesn't contain particle values
    extract_box_data = False  # start of file doesn't contain box dimension

    box_volume = 1
    box_dimensions = []  # to store side lengths of box for period boundary adjustment
    com_positions = np.zeros((N_molecules, 3))
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
            if int(particle_values[2]) == int(centre):  # central particle
                com_positions[int(particle_values[1]) - 1, :] = particle_values[3:6]

    data_file.close()  # close data_file for time step t
    volume_values[i] = box_volume

    if time != 0:  # GIVE PREVIOUS DISPLACEMENT
        extra_displacement += periodic_bc_displacement(
            com_positions, previous_positions, box_dimensions
        )

    if equilibrium_flag:  # MEASURE ONGOING RMS DISPLACEMENT
        rms_disp_values[run_num, i - run_origin, :] = rms_displacement(
            com_positions + extra_displacement,  # takes current values
            initial_sample,  # reset for each eq run
            use_vector=DIRECTIONAL_COEFF,
        )

    if time in sampling_times:  # MEASURE SAMPLING POINTS
        print("T = " + str(time) + "/" + str(run_time))
        where_output = np.where(sampling_times == time)
        indices = (where_output[0][0], where_output[1][0])
        if indices[1] == 0:  # start of sampling period
            initial_sample = com_positions
            sampled_vol_values[indices[0]] = box_volume

            extra_displacement = np.zeros_like(com_positions)  # reset for each sample

            run_origin = i  # so ongoing measurement starts from zero each time
            equilibrium_flag = True
        else:  # end of sampling period
            # print(extra_displacement) # useful to check you aren't getting silly values/multiple crossings
            sampled_rms = rms_displacement(
                com_positions + extra_displacement,
                initial_sample,
                use_vector=DIRECTIONAL_COEFF,
            )  # initial sample taken from previous iteration in if clause
            sampled_D_values[indices[0], :] = sampled_rms / (6 * equilibrium_time)
            # D value for i-th equillibration period

            run_num += 1
            equilibrium_flag = False

        if time == sampling_times[indices[0] + 1, 0] and REPEAT_CORRECTION:
            # The next sample starts as this one ends (ie due to zero shrinking time)
            initial_sample = com_positions
            sampled_vol_values[indices[0]] = box_volume
            extra_displacement = np.zeros_like(com_positions)  # reset for each sample

            run_origin = i  # so ongoing measurement starts from zero each time
            equilibrium_flag = True

    previous_positions = com_positions
    # print("T = " + str(time) + "/" + str(run_time))


# print(sampled_D_values)
# # NaN values correspond to a misalignment with dump frequency and the ends of each equillibration run
# print(sampled_vol_values)

plot_list = range(0, run_num_tot, 1)  # runs to plot

sampled_vol_frac = vol_frac(sampled_vol_values, mol_length, N_molecules)

fig, axs = plt.subplots(nrows=1, ncols=len(plot_list), sharey=True, figsize=(10, 5))
for plot_index, data_index in enumerate(plot_list):
    axs[plot_index].set_title(
        r"$\phi =$" + "{:.2f}".format(sampled_vol_frac[data_index])
    )
    # print(rms_disp_values[data_index, :, 0])
    rms_disp_values[data_index, 0, 0] = rms_disp_values[data_index, 1, 0]  # remove nan
    if np.isnan(rms_disp_values[data_index, -1, 0]):  # remove nan at end of array
        rms_disp_values[data_index, -1, :] = rms_disp_values[data_index, -2, :]

    eq_time_values = np.array(eq_range)
    eq_time_values[0] = eq_time_values[1]  # remove zero so log can be evaluated

    slope_x, intercept_x, r_value_x, p_value_x, std_err_x = linregress(
        np.log10(eq_time_values), np.log10(rms_disp_values[data_index, :, 0])
    )  # consider x axis for purpose of this
    slope_y, intercept_y, r_value_y, p_value_y, std_err_y = linregress(
        np.log10(eq_time_values), np.log10(rms_disp_values[data_index, :, 1])
    )  # consider x axis for purpose of this

    plot_best_fit = False

    print(
        "X: For vol frac = "
        + "{:.2f}".format(sampled_vol_frac[data_index])
        + ", x_slope = "
        + "{:.2f}".format(slope_x)
        + ", y_slope = "
        + "{:.2f}".format(slope_y)
        + ", intercept ratio = "
        + "{:.2f}".format(10 ** intercept_x / 10 ** intercept_y)
    )

    for j in range(dimension_num):
        if plot_index == 0:  # for legend
            axs[plot_index].loglog(
                eq_range, rms_disp_values[data_index, :, j], label=axis_labels[j],
            )
            if plot_best_fit == True and j == 2:  # only needs to be plotted once
                axs[plot_index].plot(
                    eq_time_values,
                    (eq_time_values ** slope_x) * (10 ** intercept_x),
                    label="Best fit (x)",
                    linestyle="dashed",
                )
                axs[plot_index].plot(
                    eq_time_values,
                    (eq_time_values ** slope_y) * (10 ** intercept_y),
                    label="Best fit (y)",
                    linestyle="dashed",
                )
        else:  # no legend entries
            axs[plot_index].loglog(eq_range, rms_disp_values[data_index, :, j])

            if plot_best_fit == True and j == 2:
                axs[plot_index].plot(
                    eq_time_values,
                    (eq_time_values ** slope_x) * (10 ** intercept_x),
                    linestyle="dashed",
                )
                axs[plot_index].plot(
                    eq_time_values,
                    (eq_time_values ** slope_y) * (10 ** intercept_y),
                    linestyle="dashed",
                )

axs[int(len(plot_list) / 2)].set_xlabel(
    "Time (Arbitrary Units)"
)  # use median of plot_list
axs[0].set_ylabel("RMS displacement")
fig.legend(loc="center right")
plt.savefig("rms_displacement_runwise2.png")
plt.show()

marker_style = ["x", "1", "+"]
for i in range(dimension_num):
    plt.plot(
        sampled_vol_frac,
        sampled_D_values[:, i],
        "x",
        label=axis_labels[i],
        marker=marker_style[i],
    )
plt.ylabel("Diffusion Coefficient")
plt.xlabel("Volume Fraction")
plt.legend()
plt.savefig("order_vs_diffusion_with_bc.png")
plt.show()

print("Volume fraction = " + str(sampled_vol_frac))
print("D_x/D_y = " + str(sampled_D_values[:, 0] / sampled_D_values[:, 1]))
