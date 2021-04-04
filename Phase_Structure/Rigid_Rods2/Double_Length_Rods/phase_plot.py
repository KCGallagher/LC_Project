import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d  # for rolling average
import pickle


plt.rcParams.update({"font.size": 12})  # for figures to go into latex at halfwidth


def vol_frac(volume_data):
    """Returns array of volume fraction data from volume array"""
    return np.reciprocal(volume_data) * (16 * N * (np.pi * 0.5 ** 2))
    # See OneNote details of form to use here


loop_var_name = "mix_steps"  # user defined!

file_name = "log.lammps"
start_lines = []
end_lines = []
blank_lines = []
# this gives the number of blank lines encountered by each Loop line

data_file = open(file_name, "r")
blank_lines_count = 0
for i, line in enumerate(data_file):
    if line.isspace():
        blank_lines_count += 1  # running count of num of blank lines
    if (
        "variable " + loop_var_name in line
    ):  # to extract independant variable values of loop variable
        loop_var_values = []
        for t in line.split():  # separate by whitespace
            try:
                loop_var_values.append(float(t))
            except ValueError:
                pass  # any non-floats in this line are ignored
    if "variable N" in line:  # to extract independant variable value of N
        loop_var_values = []
        for t in line.split():  # separate by whitespace
            try:
                N = float(t)
            except ValueError:
                pass  # any non-floats in this line are ignored

    if "Step Time" in line:  # always preceeds data section, can ammend searched string
        start_lines.append(i)  # start of final data readout
    if "Loop time" in line:
        end_lines.append(i)  # end of final data readout
        blank_lines.append(
            blank_lines_count
        )  # end of final data readout without blank lines

# loop_var_values = [5000, 10000, 15000]  # added to match temporarly fix

if not 2 * len(loop_var_values) == len(start_lines):
    print(
        "Warning: Number of loop variable values does not match the number of equillibrium runs. "
        + "Check whether you are reading in the correct loop variable in line 16"
    )
    # print("Number of loop variable values: " + str(len(loop_var_values)))
    # print("Number of eq runs: " + str(len(start_lines)))

last_line = i  # last line number in file
tot_blank_lines = blank_lines_count  # total blank lines in file
blank_lines_left = [
    tot_blank_lines - b for b in blank_lines
]  # blank lines after each 'Loop' line
end_lines_adj = [
    e_i + b_i for e_i, b_i in zip(end_lines, blank_lines_left)
]  # sum two lists
"""skip_footer doesn't count blank lines, but the iteraction above does. Therefore, we add in the number of 
blank lines below this end line, to give an adjusted end line that accounts for these extra blank lines"""

# print(start_lines, end_lines, last_line)
# print(start_lines, end_lines_adj, last_line)
data_file.close()
# This considers the last data output only

"""So it appears that each line is deleted from this as it is read. I have no idea why this is, but 
until I manage to fix that issue, I will close and reopen the file if necessary at this point"""

final_values = np.zeros(
    (4, int(len(start_lines) / 2))
)  # in form Temp/Press/PotEng/Volume
# These are the values at equillibrium. Currently output as mean values

total_loop_var = 0
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

colors = plt.cm.viridis(np.linspace(0, 1, int((len(start_lines) + 3) / 2)))

for i in range(1, len(start_lines), 2):
    # taking every other i to only get equillibrium values; each N has a thermalisation and an equillibrium run
    j = int(
        (i - 1) / 2
    )  # giving range from 0 upwards in integer steps to compare to N_list
    data = np.genfromtxt(
        fname=file_name,
        names=True,
        skip_header=start_lines[i],
        skip_footer=last_line - end_lines_adj[i] + 1,
        comments=None,
    )

    total_loop_var += loop_var_values[j]

    end_index = int(len(data["Press"]))
    start_index = int(0.9 * end_index)
    # selecting only final 10% of data to average, so sys has reached equilibrium

    final_values[0, j] = loop_var_values[j]
    final_values[1, j] = np.mean(data["Press"][start_index:end_index])
    final_values[2, j] = np.mean(data["PotEng"][start_index:end_index])
    final_values[3, j] = np.mean(data["Volume"][start_index:end_index])

    vol_frac_value = vol_frac(final_values[3, j])

    plt.plot(
        data["Step"] - np.min(data["Step"]),  # so time starts from zero
        # data["Press"],
        uniform_filter1d(data["Press"], size=int(5e2)),  # rolling average
        label=r"$\phi = $" + "{:.2f}".format(vol_frac_value),
        color=colors[j],
    )


vol_frac_data = vol_frac(final_values[3, :])
print("Volume Fractions: " + str(vol_frac_data))
pickle.dump(vol_frac_data, open("volume_fractions.p", "wb"))

print(data.dtype.names)

plt.xlabel("Time (Arbitrary Units)")
plt.ylabel("Pressure (Natural Units)")
# plt.title("Evolution of Thermodynamic Variables at different " + loop_var_name)
# plt.legend(
#     loc=6, bbox_to_anchor=(0.75, 0.8), labelspacing=-2.5,
# )
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc="best")
plt.tight_layout()
plt.savefig("pressureplot_frac.eps")
plt.show()

plt.plot(
    vol_frac_data,  # loop_var_values,
    final_values[2, :] / np.amax(final_values[2, :]),
    label="Normalised Internal energy",
    linestyle="",
    marker="o",
)
plt.plot(
    vol_frac_data,  # loop_var_values,
    final_values[1, :] / np.amax(final_values[1, :]),
    label="Normalised Pressure",
    linestyle="",
    marker="x",
)

plt.xlabel("Volume Fraction")
plt.ylabel("Normalised Thermodynamic Variable")
plt.legend()
plt.title("Phase Plot for " + str(int(N)) + " Rigid Rods")
plt.savefig("phaseplot_frac.png")
plt.show()

