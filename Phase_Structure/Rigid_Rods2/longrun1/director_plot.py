import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d  # for rolling average

file_root = "output_T_0.5__time_"  # two underscores to match typo in previous code
sampling_freq = 1  # only samples one in X files (must be integer)

# READ PARAMETER VALUES FROM LOG FILE

file_name = "log.lammps"
log_file = open(file_name, "r")
tot_mix_time = 0

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
                tot_mix_time += int(t)
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

run_time = tot_mix_time + run_num * equilibrium_time
time_range = range(0, int(run_time), int(dump_interval * sampling_freq))
print(
    "N_molecules, run_time, dump_interval = "
    + str((N_molecules, run_time, dump_interval))
)

# time_range = range(0, 500, 100)  # FOR SIMPLICITY IN TESTING

# DEFINE FUNCTION TO FIND DIRECTOR FROM OUTPUT DATA
def order_param(data):
    """Input data in array of size Molecule Number x 3 x 3

    Order param is defined as ave((3cos^2(theta)-1)/2) where theta is the angle between
    the molecule director and mean director

    Input data will be rod_positions array which stores input data   
    First index gives molecule number
    Second index gives particle number within molecule (first/last)
    Third index gives the component of the position (x,y,z)
    """
    directors = data[:, 1, :] - data[:, 0, :]  # director vector for each molecule
    mean_director = np.mean(directors, axis=0)
    norm_mean_director = mean_director / np.linalg.norm(mean_director)
    cosine_values = np.zeros_like(directors)
    for index, vector in enumerate(directors):
        cosine_values[index] = np.dot(
            norm_mean_director, (vector / np.linalg.norm(vector))
        )
    order_param = np.mean((3 * np.square(cosine_values) - 1) / 2)
    return order_param


# READ MOLECULE POSITIONS

order_param_values = np.zeros(len(time_range))
for i, time in enumerate(time_range):  # interate over dump files
    data_file = open(file_root + str(time) + ".dump", "r")
    extract_data = False  # start of file doesn't contain particle values

    rod_positions = np.zeros((N_molecules, 2, 3))
    """Indices are Molecule Number/ First (0) or Last (1) atom,/ Positional coord index"""

    for line in data_file:
        if "ITEM: ATOMS" in line:  # to start reading data
            extract_data = True
            continue  # don't attempt to read this line

        if extract_data:
            # each line is in the form "id mol type x y z vx vy vz"
            particle_values = []
            for t in line.split():  # separate by whitespace
                try:
                    particle_values.append(float(t))
                except ValueError:
                    pass  # any non-floats in this line are ignored

            # Save positional coordatinates of end particles
            if int(particle_values[2]) == 1:  # first particle in molecule
                rod_positions[int(particle_values[1]) - 1, 0, :] = particle_values[3:6]
            if int(particle_values[2]) == 10:  # last particle in molecule
                rod_positions[int(particle_values[1]) - 1, 1, :] = particle_values[3:6]

    data_file.close()  # close data_file for time step t
    order_param_values[i] = order_param(rod_positions)  # evaluate order param at time t
    print("T = " + str(time) + "/" + str(run_time))

plt.plot(time_range, abs(order_param_values))
plt.plot(
    time_range,
    uniform_filter1d(abs(order_param_values), size=int(1e2)),
    linestyle="--",
)
plt.xlabel("Time (arbitrary units)")
plt.ylabel("Order Parameter")
plt.title("Evolution of Order Parameter")
#plt.savefig("director_evo.png")
plt.show()
