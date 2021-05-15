import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from phase_plot import vol_frac

file_root = "output_T_0.5_time_"
sampling_freq = 40  # only samples one in X files (must be integer) #30
plotting_freq = 1  # only plots on in X of the sampled distributions

plt.rcParams.update({"font.size": 13})  # for figures to go into latex at halfwidth

# ... Same data extraction procedure as phase_plot.py ...


def angle_dist(data, remove_split_mol=True):
    """Input data in array of size Molecule Number x 3 x 3

    Input data will be rod_positions array which stores input data   
    First index gives molecule number
    Second index gives particle number within molecule (1st/mid/last)
    Third index gives the component of the position (x,y,z)

    Allows option to remove molecules that are split across boundaries

    """

    if remove_split_mol:
        molecules_removed = 0
        # where molecules span simulation region boundaries, replace with nans
        for i in range(N_molecules):
            if np.linalg.norm(data[i, 2, :] - data[i, 0, :]) > (mol_length + 0.5):
                data[i, :, :].fill(np.nan)
                molecules_removed += 1
                # remove data for molecules that are longer than expected
                # this is due to them spanning the edges of the simulation region
        print("Number of molecules removed is : " + str(molecules_removed))

    rod_1 = data[:, 1, :] - data[:, 0, :]  # director vector for first arm
    norm_rod_1 = rod_1 / np.linalg.norm(rod_1, axis=1).reshape(-1, 1)
    rod_2 = data[:, 1, :] - data[:, 2, :]  # director vector for second arm
    # note this is defined so the two vectors point away from the centre
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

    # ... Same positional extraction as correlation_plot.py ...

    data_file.close()  # close data_file for time step t
    volume_values[i] = box_volume

    angle_data = angle_dist(rod_positions)
    angle_mean_values[i] = np.mean(angle_data)  # evaluate order param at time t

    angle_data = np.where(
        angle_data < 0.8, angle_data, np.nan
    )  # remove spurious high values

    tot_plot_num = len(time_range) // plotting_freq
    colors = plt.cm.viridis(np.linspace(0, 1, tot_plot_num))
    if i % plotting_freq == 0 and time != 0:
        print(time)
        sns.kdeplot(
            angle_data,
            color=colors[i // plotting_freq - 1],
            bw_adjust=0.1,  # adjusts smoothing (default is 1)
            # gridsize=50,
            alpha=1,  # adjusts transparency
        )

    print("T = " + str(time) + "/" + str(run_time))

sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=0, vmax=run_time))
cbar = plt.colorbar(sm)
cbar.ax.set_ylabel("Number of Time Steps", rotation=270, labelpad=15)

plt.title("Evolution of angle distribution")
plt.xlim([-1, 1])
plt.xlabel(r"Nunchuck Angle ($cos(\theta)$)")
plt.ylabel("Normalised Frequency")
plt.savefig("nun_fr_angledist.eps")
plt.show()
