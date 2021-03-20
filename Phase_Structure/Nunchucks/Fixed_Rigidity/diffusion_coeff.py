"""Traditional method to calculate the pair-wise orientational order correlation function
Uses the end-to-end molecule vector as the director"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d  # for rolling average
from phase_plot import vol_frac

FILE_ROOT = "output_T_0.5_time_"  # two underscores to match typo in previous code
SAMPLING_FREQ = 1  # only samples one in X files (must be integer)

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
time_range = np.arange(0, int(run_time), int(dump_interval * SAMPLING_FREQ))
print(
    "N_molecules, run_time, dump_interval = "
    + str((N_molecules, run_time, dump_interval))
)


def find_separation(pos1, pos2, box_dim):
    """Finds separation between two positions

    This method finds the minimum separation, accounting for the periodic BC
    pos1, pos2 are the position vectors of the two points
    box_dim is a vector (of equal length) giving the dimensions of the simulation region"""
    separation = pos1 - pos2
    for i in range(len(pos1)):  # should be 3 dimensional
        if np.abs(pos1[i] - pos2[i]) > box_dim[i] / 2:
            # use distance to ghost instead
            separation[i] = box_dim[i] - np.abs(pos1[i] - pos2[i])

    return np.linalg.norm(separation)


def rms_displacement(pos_t, pos_0, box_dim):
    """Input data in array of size Molecule Number x 3, and list of box_dim

    Input data will be com_positions array which stores input data   
    First index gives molecule number
    Second index gives the component of the position (x,y,z)

    Returns rms displacement from initial displacement"""

    rms_value = np.linalg.norm((pos_t - pos_0))

    return np.mean(rms_value)


# READ MOLECULE POSITIONS

rms_disp_values = np.zeros(len(time_range))
volume_values = np.full(len(time_range), np.nan)  # new array of NaN
for i, time in enumerate(time_range):  # interate over dump files
    data_file = open(FILE_ROOT + str(time) + ".dump", "r")
    extract_atom_data = False  # start of file doesn't contain particle values
    extract_box_data = False  # start of file doesn't contain box dimension

    box_volume = 1
    box_dimensions = []  # to store side lengths of box for period boundary adjustment
    com_positions = np.zeros((N_molecules, 3, 3))
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
            if int(particle_values[2]) == int(centre):  # central particle
                com_positions[int(particle_values[1]) - 1, :] = particle_values[3:6]

    data_file.close()  # close data_file for time step t
    volume_values[i] = box_volume

    if time == 0:
        initial_positions = com_positions
        rms_disp_values[0] = np.nan
    else:
        rms_disp_values[i] = rms_displacement(
            com_positions, initial_positions, box_dimensions
        )  # evaluate <x^2> at time t

    print("T = " + str(time) + "/" + str(run_time))


time_range[0] = 1  # avoid divide by zero error, will be ignored anyway
diffusion_coeff_values = (
    (1 / 6) * rms_disp_values * np.reciprocal(time_range, dtype="float64")
)
print("Mean Diffussion Coefficients: " + str(np.nanmean(diffusion_coeff_values)))

plt.plot(time_range, rms_disp_values)
plt.xlabel("Time (arbitrary units)")
plt.ylabel("Diffusion Coefficient")
plt.savefig("rms_displacement.png")
plt.show()

fig, ax1 = plt.subplots()

color = "tab:red"
ax1.set_xlabel("Time (arbitrary units)")
ax1.set_ylabel("Diffusion Coefficient", color=color)
ax1.plot(
    time_range, diffusion_coeff_values, color=color,
)
ax1.tick_params(axis="y", labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = "tab:blue"
ax2.set_ylabel(
    "Volume Fraction", color=color
)  # we already handled the x-label with ax1
ax2.plot(time_range, vol_frac(volume_values, mol_length, N_molecules), color=color)
ax2.tick_params(axis="y", labelcolor=color)

plt.title("Evolution of Diffusion Coefficient")
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig("order_and_diffusion.png")
plt.show()

plt.plot(vol_frac(volume_values), diffusion_coeff_values, "rx")
plt.ylabel("Order Parameter")
plt.xlabel("Volume Fraction")
plt.savefig("order_vs_diffusion.png")
plt.show()

