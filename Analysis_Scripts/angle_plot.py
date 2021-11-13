"""Plots the distribution of opening angles in a sample.

This script records the opening angle (i.e. the angle between the two arms of 
the nunchuck) for all molecules in the sample. It should be noted that molecules whose 
centres span the boundaries of the simulation region are not included in this sample
as the measurement of their angles is typically prone to error; this is typically a 
small number of the sample and the total count of these it output for the user.

It returns a plot of kernel densities for the angle disctribution over time - typically
this may indicate the formation of a preferential angle from an isotropic distribution.
This may be more obvious in the evolution of mean angle over time.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from scipy.ndimage import uniform_filter1d  # for rolling average
from phase_plot import vol_frac

file_root = "output_T_0.5_time_"
sampling_freq = 60  # only samples one in X files (must be integer) #30
plotting_freq = 1  # only plots on in X of the sampled distributions

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
        for t in line.split():  # separate by whitespace
            try:
                mol_length = int(t)
            except ValueError:
                pass  # any non-floats in this line are ignored

    if "dump" and "all custom" in line:  # to extract time interval for dump
        for t in line.split():  # separate by whitespace
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
time_range = range(0, int(run_time), int(dump_interval * sampling_freq))
time_range = range(
    int(run_time), 0, int(dump_interval * sampling_freq * -1)
)  # reversed order
print(
    "N_molecules, run_time, dump_interval = "
    + str((N_molecules, run_time, dump_interval))
)


# time_range = range(0, 3300000, 100000)  # FOR SIMPLICITY IN TESTING


def angle_dist(data, remove_split_mol=True):
    """Input data in array of size Molecule Number x 3 x 3

    Input data will be rod_positions array which stores input data   
    First index gives molecule number
    Second index gives particle number within molecule (1st/mid/last)
    Third index gives the component of the position (x,y,z)

    Allows option to remove molecules that are split across the boundaries of the box

    """

    if remove_split_mol:
        molecules_removed = 0
        # where molecule end-to-end length is greater than expected, replace with nans
        for i in range(N_molecules):
            if np.linalg.norm(data[i, 2, :] - data[i, 0, :]) > (mol_length + 0.5):
                data[i, :, :].fill(np.nan)
                molecules_removed += 1
                # remove data for molecules that are longer than expected (ie 10 units)
                # this is due to them spanning the edges of the simulation region
        print("Number of molecules removed is : " + str(molecules_removed))

    rod_1 = data[:, 1, :] - data[:, 0, :]  # director vector for first end of molecule
    norm_rod_1 = rod_1 / np.linalg.norm(rod_1, axis=1).reshape(-1, 1)
    rod_2 = data[:, 1, :] - data[:, 2, :]  # director vector for second end of molecule
    # note this is defined so the two vectors point away from the centre of the molecule
    norm_rod_2 = rod_2 / np.linalg.norm(rod_2, axis=1).reshape(-1, 1)

    angle_values = np.sum(norm_rod_1 * norm_rod_2, axis=1)
    angle_values = angle_values[~np.isnan(angle_values)]  # remove nans
    return angle_values


# READ MOLECULE POSITIONS

angle_mean_values = np.zeros(len(time_range))
volume_values = np.full(len(time_range), np.nan)  # new array of NaN
for i, time in enumerate(time_range):  # interate over dump files
    data_file = open(file_root + str(time) + ".dump", "r")
    extract_atom_data = False  # start of file doesn't contain particle values
    extract_box_data = False  # start of file doesn't contain box dimension

    box_volume = 1
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
            box_dimension = []
            for d in line.split():  # separate by whitespace
                try:
                    box_dimension.append(float(d))
                except ValueError:
                    pass  # any non-floats in this line are ignored
            box_volume *= box_dimension[1] - box_dimension[0]
            # multiply box volume by length of this dimension of box

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
            centre = (mol_length + 1) / 2  # for odd length molecules only
            if int(particle_values[2]) == int(centre - 1):
                rod_positions[int(particle_values[1]) - 1, 0, :] = particle_values[3:6]
            if int(particle_values[2]) == int(centre):  # central particle
                rod_positions[int(particle_values[1]) - 1, 1, :] = particle_values[3:6]
            if int(particle_values[2]) == int(centre + 1):
                rod_positions[int(particle_values[1]) - 1, 2, :] = particle_values[3:6]

    data_file.close()  # close data_file for time step t
    volume_values[i] = box_volume

    angle_data = angle_dist(rod_positions, remove_split_mol=True)
    angle_mean_values[i] = np.mean(angle_data)  # evaluate order param at time t

    angle_data = np.where(
        angle_data < 0.8, angle_data, np.nan
    )  # remove spurious high values

    tot_plot_num = len(time_range) // plotting_freq
    colors = plt.cm.gnuplot(np.linspace(0, 1, tot_plot_num))
    if i % plotting_freq == 0 and time != 0:
        sns.kdeplot(
            np.arccos(angle_data) * (180 / np.pi),
            color=colors[i // plotting_freq],  # -1 for normal order plotting
            bw_adjust=0.1,  # adjusts smoothing (default is 1)
            # gridsize=5000,
            alpha=1,  # adjusts transparency
        )

    print(
        "T = "
        + str(time)
        + "/"
        + str(run_time)
        + ", Mean = "
        + str(np.nanmean(angle_data))
    )

sm = plt.cm.ScalarMappable(cmap=cm.gnuplot_r, norm=plt.Normalize(vmin=0, vmax=run_time))
cbar = plt.colorbar(sm)
cbar.ax.set_ylabel("Number of Time Steps", rotation=270, labelpad=15)

# plt.title("Evolution of angle distribution")
plt.xlim([60, 180])
plt.xlabel(r"Nunchuck Angle ($\theta$)")
plt.ylabel("Normalised Frequency")
plt.savefig("nun_fr_angledist.svg")
plt.show()

plt.plot(time_range, angle_mean_values)
plt.plot(
    time_range, uniform_filter1d(angle_mean_values, size=int(10)), linestyle="--",
)
plt.xlabel("Time (arbitrary units)")
plt.ylabel(r"Mean Angle ($cos(\theta)$)")
plt.title("Evolution of Mean Angle")
# plt.savefig("angle_mean.png")
plt.show()
