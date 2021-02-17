import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d  # for rolling average
from phase_plot import vol_frac

file_root = "output_T_0.5_time_"
sampling_freq = 1  # only samples one in X files (must be integer)
plotting_freq = 50  # only plots on in X of the sampled distributions

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
print(
    "N_molecules, run_time, dump_interval = "
    + str((N_molecules, run_time, dump_interval))
)


# time_range = range(0, 3300000, 100000)  # FOR SIMPLICITY IN TESTING


def angle_dist(data):
    """Input data in array of size Molecule Number x 4 x 3

    Input data will be rod_positions array which stores input data   
    First index gives molecule number
    Second index gives particle number within molecule (1/5/6/10)
    Third index gives the component of the position (x,y,z)

    """
    rod_1 = data[:, 1, :] - data[:, 0, :]  # director vector for first end of molecule
    norm_rod_1 = rod_1 / np.linalg.norm(rod_1, axis=1).reshape(-1, 1)
    rod_2 = data[:, 3, :] - data[:, 2, :]  # director vector for second end of molecule
    norm_rod_2 = rod_2 / np.linalg.norm(rod_2, axis=1).reshape(-1, 1)

    angle_values = np.sum(norm_rod_1 * norm_rod_2, axis=1)
    return angle_values


# READ MOLECULE POSITIONS

angle_mean_values = np.zeros(len(time_range))
volume_values = np.full(len(time_range), np.nan)  # new array of NaN
for i, time in enumerate(time_range):  # interate over dump files
    data_file = open(file_root + str(time) + ".dump", "r")
    extract_atom_data = False  # start of file doesn't contain particle values
    extract_box_data = False  # start of file doesn't contain box dimension

    box_volume = 1
    rod_positions = np.zeros((N_molecules, 4, 3))
    """Indices are Molecule Number; Atom number 1/5/6/10 ; Positional coord index"""

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

            # Save positional coordatinates of end particles
            if int(particle_values[2]) == 1:  # first particle in first rod
                rod_positions[int(particle_values[1]) - 1, 0, :] = particle_values[3:6]
            if int(particle_values[2]) == 5:  # last particle in first rod
                rod_positions[int(particle_values[1]) - 1, 1, :] = particle_values[3:6]
            if int(particle_values[2]) == 6:  # first particle in second rod
                rod_positions[int(particle_values[1]) - 1, 2, :] = particle_values[3:6]
            if int(particle_values[2]) == 10:  # last particle in second rod
                rod_positions[int(particle_values[1]) - 1, 3, :] = particle_values[3:6]

    data_file.close()  # close data_file for time step t
    volume_values[i] = box_volume
    angle_mean_values[i] = np.mean(
        angle_dist(rod_positions)
    )  # evaluate order param at time t

    tot_plot_num = len(time_range) // plotting_freq
    colors = plt.cm.cividis(np.linspace(0, 1, tot_plot_num))
    if i % plotting_freq == 0 and time != 0:
        if i == plotting_freq or i == tot_plot_num * plotting_freq:
            # label only start and end points
            plt.hist(
                angle_dist(rod_positions),
                density=True,
                histtype="step",
                color=colors[i // plotting_freq - 1],
                label=("T = " + str(int(time))),
            )
        else:
            plt.hist(
                angle_dist(rod_positions),
                density=True,
                histtype="step",
                color=colors[i // plotting_freq - 1],
            )

    print("T = " + str(time) + "/" + str(run_time))

plt.title("Evolution of angle distribution over time")
plt.xlabel(r"Mean Angle ($cos(\theta)$)")
plt.ylabel("Normalised Frequency")
plt.legend()
plt.show()

plt.plot(time_range, angle_mean_values)
plt.plot(
    time_range, uniform_filter1d(angle_mean_values, size=int(10)), linestyle="--",
)
plt.xlabel("Time (arbitrary units)")
plt.ylabel(r"Mean Angle ($cos(\theta)$)")
plt.title("Evolution of Mean Angle")
plt.savefig("angle_mean.png")
plt.show()
