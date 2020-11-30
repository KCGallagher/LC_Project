import numpy as np
import matplotlib.pyplot as plt

loop_var_name = "mix_steps"

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
    ):  # to extract independant variable values of N (or T etc)
        index_values = []
        for t in line.split():  # separate by whitespace
            try:
                index_values.append(float(t))
            except ValueError:
                pass  # any non-floats in this line are ignored

    if "Step Time" in line:  # always preceeds data section, can ammend searched string
        start_lines.append(i)  # start of final data readout
    if "Loop time" in line:
        end_lines.append(i)  # end of final data readout
        blank_lines.append(
            blank_lines_count
        )  # end of final data readout without blank lines

if not 2 * len(index_values) == len(start_lines):
    print(
        "Warning: Number of loop variable values does not match the number of equillibrium runs. "
        + "Check whether you are reading in the correct loop variable in line 16"
    )

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

final_values = np.zeros((3, int(len(start_lines) / 2)))  # in form Temp/Press/PotEng
# These are the values at equillibrium. Currently output as mean values

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
    # print(start_lines[i], last_line - end_lines[i])
    # print(data)
    plt.plot(
        data["Step"] - np.min(data["Step"]),  # so time starts from zero
        data["Press"],
        label=loop_var_name + "= " + str(index_values[j]),
    )
    final_values[0, j] = index_values[j]
    final_values[1, j] = np.mean(data["Press"])
    final_values[2, j] = np.mean(data["PotEng"])

print(data.dtype.names)

plt.xlabel("Time Step")
plt.ylabel("Natural Units")
plt.title("Evolution of Thermodynamic Variables at different " + loop_var_name)
plt.legend()
plt.show()

plt.plot(
    index_values,
    final_values[1, :] / np.amax(final_values[1, :]),
    label="Normalised Pressure",
)
plt.plot(
    index_values,
    final_values[2, :] / np.amax(final_values[2, :]),
    label="Normalised Internal energy",
)
plt.xlabel("Value of " + loop_var_name)
plt.ylabel("Normalised Thermodynamic Variable")
plt.legend()
plt.show()

