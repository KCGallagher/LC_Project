"""Additional approaches to calcuating the pair-wise orientational order 
correlation function, using different director vectors"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from scipy.ndimage import uniform_filter1d  # for rolling average
from phase_plot import vol_frac

FILE_ROOT = "output_T_0.5_time_"  # two underscores to match typo in previous code
SAMPLING_FREQ = 20  # only samples one in X files (must be integer)
SEPARATION_BIN_NUM = 30  # number of bins for radius dependance pair-wise correlation

mol_length = 15

DIRECTOR_METHOD = "molecule"
# Options are molecule/arm/bisector/normal


plt.rcParams.update({"font.size": 13})  # for figures to go into latex at halfwidth

# ... Same data extraction procedure as phase_plot.py ...


def find_director(data, method="molecule"):
    """Obtains director from molecule positions, through a variety of methods
    
    First index gives molecule number
    Second index gives particle number within molecule 
        Corresponds to: start of director/ centre/ end of director
    Third index gives the component of the position (x,y,z)

    'molecule' calculates director between ends of the molecule; 
    'arm' calculates director along first arm of molecule; 
    'bisector' gives the director along the bisector of the join angle; 
    'normal' gives the bisector out of the plane of the molecule
    """

    if method == "molecule":
        return data[:, 2, :] - data[:, 0, :]

    elif method == "arm":
        return data[:, 1, :] - data[:, 0, :]

    elif method == "bisector":
        midpoints = 0.5 * (data[:, 2, :] + data[:, 0, :])
        return data[:, 1, :] - midpoints

    elif method == "normal":
        arm1 = data[:, 1, :] - data[:, 0, :]
        arm2 = data[:, 2, :] - data[:, 1, :]
        return np.cross(arm1, arm2)

    else:
        raise ValueError("Unknown argument to find_director()")


def find_angle(vec1, vec2):
    """Finds angle between two vectors"""
    assert len(vec1) == len(vec2), "Vectors should be the same dimension"

    vec1 = vec1 / np.linalg.norm(vec1)  # normalise vectors
    vec2 = vec2 / np.linalg.norm(vec2)

    return np.sum(vec1 * vec2)


def find_separation(pos1, pos2, box_dim):
    """Finds separation between two positions

    This method finds the minimum separation, accounting for the periodic BC
    pos1, pos2 are the position vectors of the two points
    box_dim is a vector (of equal length) giving the dim of the simulation region"""
    separation = pos1 - pos2
    for i in range(len(pos1)):  # should be 3 dimensional
        if np.abs(pos1[i] - pos2[i]) > box_dim[i] / 2:
            # use distance to ghost instead
            separation[i] = box_dim[i] - np.abs(pos1[i] - pos2[i])

    return np.linalg.norm(separation)


def eval_angle_array(data, box_dim):
    """Input data in array of size Molecule Number x 3 x 3, and list of box_dim

    Input data will be rod_positions array which stores input data   
    First index gives molecule number
    Second index gives particle number within molecule 
        Corresponds to: start of director/ centre/ end of director
    Third index gives the component of the position (x,y,z)

    Outputs N x N x 2 array, for pairwise values of separation and angle
    Be aware this may generate very large arrays
    """
    angle_array = np.full((N_molecules, N_molecules, 2), np.nan, dtype=np.float32)
    # dtype specified to reduce storgae required

    director = find_director(
        data, method=DIRECTOR_METHOD
    )  # director vector for whole of molecule

    for i in range(N_molecules - 1):
        for j in range(i + 1, N_molecules):
            # Only considering terms of symmetric matrix above diagonal
            # Separation between centres of each molecule:
            angle_array[i, j, 0] = find_separation(
                data[i, 1, :], data[j, 1, :], box_dim
            )
            # Angle between arms of molecule:
            angle_array[i, j, 1] = find_angle(director[i, :], director[j, :])
    angle_array_masked = np.ma.masked_invalid(
        angle_array[:, :, :]
    )  # mask empty values below diagonal
    return angle_array_masked


def correlation_func(data, box_dim):
    """Input data in array of size Molecule Number x 3 x 3, and list of box_dim

    Input data will be rod_positions array which stores input data   
    First index gives molecule number
    Second index gives particle number within molecule (first/last)
    Third index gives the component of the position (x,y,z)

    Returns array of correlation data at each radius"""

    angle_array = eval_angle_array(data, box_dim)
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
        )  # evaluate 2nd order legendre polynomial

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
    box_dimensions = []  # to store side lengths of box for period boundary adjustment
    rod_positions = np.zeros((N_molecules, 3, 3))
    """Indices are Molecule Number; Atom number 1st/mid/last ; Positional coord index"""

    # ... Same position extraction procedure as correlation_plot.py

    data_file.close()  # close data_file for time step t
    volume_values[i] = box_volume
    separation_bins, correlation_data = correlation_func(
        rod_positions, box_dimensions
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
image_name = "correlation_func_test" + str(DIRECTOR_METHOD) + ".png"
plt.savefig(image_name)
plt.show()
