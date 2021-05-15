import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import uniform_filter1d  # for rolling average
from phase_plot import vol_frac

file_root = "output_T_0.5_time_"
sampling_freq = 5  # only samples one in X files (must be integer)
plotting_freq = 5  # only plots on in X of the sampled distributions

plt.rcParams.update({"font.size": 13})  # for figures to go into latex at halfwidth


def order_param(data):
    """Input data in array of length Molecule Number 

    Input data will be rod_positions array which stores input data   
    First index gives molecule number
    Third index gives the component of the position of CoM (x,y,z)

    Method for calculation given by Polson (1997) 10.1103/PhysRevE.56.R6260
    """
    k = 2 * np.pi / 10  # divide by length of molecule
    exp_values = np.exp(1j * k * data)
    return np.abs(np.sum(exp_values)) / N_molecules


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


def CoM_dist(data):
    """Input data in array of size Molecule Number x 3

    Input data will be rod_positions array which stores input data   
    First index gives molecule number
    Second index gives particle number within molecule (1/5/6/10)
    Third index gives the component of the position (x,y,z)

    """
    return data[:, 1]  # return y coordinate


# ... Same initial extraction as nematic order plot ...


# READ MOLECULE POSITIONS

volume_values = np.full(len(time_range), np.nan)  # new array of NaN
order_param_values = np.full(len(time_range), np.nan)
CoM_mean_values = []
CoM_mean_times = []
for i, time in enumerate(time_range):  # interate over dump files
    data_file = open(file_root + str(time) + ".dump", "r")
    extract_atom_data = False  # start of file doesn't contain particle values
    extract_box_data = False  # start of file doesn't contain box dimension

    # ... same extraction as nematic order plot...

    box_volume = 1
    rod_positions = np.zeros((N_molecules, 3))
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

            # Save positional coordatinates of centre particle
            if int(particle_values[2]) == 5:
                rod_positions[int(particle_values[1]) - 1, :] = particle_values[3:6]

    data_file.close()  # close data_file for time step t
    volume_values[i] = box_volume
    order_param_values[i] = order_param(rod_positions[:, 1])

    CoM_data = rod_positions[:, 1]  # select y coordinate
    # kde_data = scipy.stats.gaussian_kde(CoM_data)

    plot_num = 0
    tot_plot_num = len(time_range) // plotting_freq
    colors = plt.cm.cividis(np.linspace(0, 1, tot_plot_num))
    if i % plotting_freq == 0 and time != 0:
        if i == plotting_freq or time >= run_time - (
            dump_interval * sampling_freq * plotting_freq
        ):

            my_kde = sns.kdeplot(
                CoM_data,
                label="T = " + str(int(time)),  # start and end times
                color=colors[i // plotting_freq - 1],
                bw_adjust=0.5,  # adjusts smoothing (default is 1)
                cut=0,
                # clip=(0, 50), # cuts off x limit
                # gridsize=50,  #adjusts points in average (default is 200)
            )
        else:
            my_kde = sns.kdeplot(
                CoM_data,
                color=colors[i // plotting_freq - 1],
                alpha=1,
                bw_adjust=0.5,
                cut=0,
                # clip=(0, 50), # cuts off x limit
                # gridsize=50
            )
            # alpha may be used to adjust transparency
        line = my_kde.lines[plot_num - 1]
        x, y = line.get_data()
        CoM_mean_values.append(max(y) - min(y))
        CoM_mean_times.append(time)  # evaluate order param at time t
        plot_num += 1
        print(max(y), min(y))

    print("T = " + str(time) + "/" + str(run_time))

plt.title("Evolution of CoM distribution over time")
plt.xlabel("Mean CoM")
plt.ylabel("Normalised Frequency")
plt.legend()
plt.savefig("CoM_dist.png")
plt.show()


fig, ax1 = plt.subplots()

color = "tab:red"
ax1.set_xlabel("Time (arbitrary units)")
ax1.set_ylabel("Smectic Order Parameter", color=color)
ax1.plot(time_range, order_param_values, color=color)
ax1.tick_params(axis="y", labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = "tab:blue"
ax2.set_ylabel(
    "Volume Fraction", color=color
)  # we already handled the x-label with ax1
ax2.plot(time_range, vol_frac(volume_values), color=color)
ax2.tick_params(axis="y", labelcolor=color)

plt.title("Evolution of Order Parameter")
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig("s_order_and_volfrac2.png")
plt.show()