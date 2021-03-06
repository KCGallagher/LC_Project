import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from scipy.ndimage import uniform_filter1d  # for rolling average
from phase_plot import vol_frac

FILE_ROOT = "output_T_0.5_time_"  # two underscores to match typo in previous code
SAMPLING_FREQ = 200  # only samples one in X files (must be integer)
SEPARATION_BIN_NUM = 20  # number of bins for radius dependance pair-wise correlation

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


# time_range = range(0, 3300000, 100000)  # FOR SIMPLICITY IN TESTING


def find_angle(vec1, vec2):
    """Finds angle between two vectors"""
    assert len(vec1) == len(vec2), "Vectors should be the same dimension"

    vec1 = vec1 / np.linalg.norm(vec1)  # normalise vectors
    vec2 = vec2 / np.linalg.norm(vec2)

    return np.sum(vec1 * vec2)


def eval_angle_array(data):
    """Input data in array of size Molecule Number x 3 x 3

    Input data will be rod_positions array which stores input data   
    First index gives molecule number
    Second index gives particle number within molecule (first/last)
    Third index gives the component of the position (x,y,z)

    Outputs N x N x 2 array, for pairwise values of separation and angle
    Be aware this may generate very large arrays
    """
    angle_array = np.full((N_molecules, N_molecules, 2), np.nan, dtype=np.float32)
    # dtype specified to reduce storgae required

    director = data[:, 2, :] - data[:, 0, :]  # director vector for whole of molecule

    for i in range(N_molecules - 1):
        for j in range(
            i + 1, N_molecules
        ):  # only consider terms of symmetric matrix above diagonal
            # Separation between centres of each molecule:
            angle_array[i, j, 0] = np.linalg.norm(data[i, 1, :] - data[j, 1, :])
            # Angle between arms of molecule:
            angle_array[i, j, 1] = find_angle(director[i, :], director[j, :])
    angle_array2 = np.ma.masked_invalid(
        angle_array[:, :, :]
    )  # mask empty values below diagonal
    return angle_array2


def correlation_func(data):
    angle_array = eval_angle_array(data)
    max_separation = np.max(angle_array[:, :, 0])

    bin_width = max_separation / SEPARATION_BIN_NUM
    separation_bins = np.linspace(0, max_separation, SEPARATION_BIN_NUM, endpoint=False)
    correlation_data = np.zeros_like(separation_bins)

    for n, radius in enumerate(separation_bins):
        # mask data outside the relevant radius range
        relevant_angles = np.ma.masked_where(
            np.logical_or(
                (angle_array[:, :, 0] < radius),
                (angle_array[:, :, 0] > (radius + bin_width)),
            ),
            angle_array[:, :, 1],  # act on angle data
        )

        legendre_polynomials = np.polynomial.legendre.legval(
            relevant_angles[:, :], [0, 0, 1]
        )

        correlation_data[n] = np.mean(legendre_polynomials)

        print("    radius = " + str(int(radius)) + "/" + str(int(max_separation)))

    return separation_bins, correlation_data


# READ MOLECULE POSITIONS

order_param_values = np.zeros(len(time_range))
volume_values = np.full(len(time_range), np.nan)  # new array of NaN
for i, time in enumerate(time_range):  # interate over dump files
    data_file = open(FILE_ROOT + str(time) + ".dump", "r")
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
            centre = (mol_length + 1) / 2
            if int(particle_values[2]) == int(centre - 1):
                rod_positions[int(particle_values[1]) - 1, 0, :] = particle_values[3:6]
            if int(particle_values[2]) == int(centre):  # central particle
                rod_positions[int(particle_values[1]) - 1, 1, :] = particle_values[3:6]
            if int(particle_values[2]) == int(centre + 1):
                rod_positions[int(particle_values[1]) - 1, 2, :] = particle_values[3:6]

    data_file.close()  # close data_file for time step t
    volume_values[i] = box_volume
    separation_bins, correlation_data = correlation_func(
        rod_positions
    )  # evaluate order param at time t

    tot_plot_num = len(time_range)
    colors = plt.cm.cividis(np.linspace(0, 1, tot_plot_num))
    if i == 0:
        continue  # don't plot this case
    plt.plot(
        separation_bins, correlation_data, color=colors[i],
    )

    print("T = " + str(time) + "/" + str(run_time))

sm = plt.cm.ScalarMappable(cmap=cm.cividis, norm=plt.Normalize(vmin=0, vmax=run_time))
cbar = plt.colorbar(sm)
cbar.ax.set_ylabel("Number of Time Steps", rotation=270, labelpad=15)

plt.title("Pairwise Angular Correlation Function")
plt.xlabel("Particle Separation")
plt.ylabel("Correlation Function")
plt.savefig("correlation_func.png")
plt.show()
