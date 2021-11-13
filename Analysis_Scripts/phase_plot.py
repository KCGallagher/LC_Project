"""Plots the development of thermodynamic variables over time

This script loops over all equillibration runs of the system, to
plot the evolution of thermodynamic variables pressure and internal
temperature of the timescale of the simulation.

This also defines a function to calculate the volume fraction from
the current volume, and molecular parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d  # for rolling average


def vol_frac(volume_data, mol_length=10, N=1000):
    """Returns array of volume fraction data from volume array

    Input list of volumes to be evaluated (volume data)
    Also takes molecule length (default 10) and number of molecules (default 1000)
    """
    return np.reciprocal(volume_data) * (mol_length * N * (np.pi * 0.56 ** 2))
    # See OneNote details of form to use here


if __name__ == "__main__":
    LOOP_VAR_NAME = "mix_steps"  # allows for loops under different names

    FILE_NAME = "log.lammps"
    start_lines = []
    end_lines = []
    blank_lines = []
    # this gives the number of blank lines encountered by each Loop line

    data_file = open(FILE_NAME, "r")
    blank_lines_count = 0
    for i, line in enumerate(data_file):
        if line.isspace():
            blank_lines_count += 1  # running count of num of blank lines
        if (
            "variable " + LOOP_VAR_NAME in line
        ):  # to extract independant variable values of loop variable
            loop_var_values = []
            for t in line.split():  # separate by whitespace
                try:
                    loop_var_values.append(float(t))
                except ValueError:
                    pass  # any non-floats in this line are ignored
        if "variable N" in line:  # to extract independant variable value of N
            for t in line.split():  # separate by whitespace
                try:
                    N = float(t)
                except ValueError:
                    pass  # any non-floats in this line are ignored
        if "variable len" in line:  # to extract length of molecule
            for t in line.split():  # separate by whitespace
                try:
                    mol_length = float(t)
                except ValueError:
                    pass  # any non-floats in this line are ignored

        if (
            "Step Time" in line
        ):  # always preceeds data section, can ammend searched string
            start_lines.append(i)  # start of final data readout
        if "Loop time" in line:
            end_lines.append(i)  # end of final data readout
            blank_lines.append(
                blank_lines_count
            )  # end of final data readout without blank lines

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

    data_file.close()

    final_values = np.zeros(
        (4, int(len(start_lines) / 2))
    )  # in form Temp/Press/PotEng/Volume
    # These are the values at equillibrium. Currently output as mean values

    total_loop_var = 0
    for i in range(1, len(start_lines), 2):
        # taking every other i to only get equillibrium values; each N has a thermalisation and an equillibrium run
        j = int(
            (i - 1) / 2
        )  # giving range from 0 upwards in integer steps to compare to N_list
        data = np.genfromtxt(
            fname=FILE_NAME,
            names=True,
            skip_header=start_lines[i],
            skip_footer=last_line - end_lines_adj[i] + 1,
            comments=None,
        )

        total_loop_var += loop_var_values[j]
        plt.plot(
            data["Step"] - np.min(data["Step"]),  # so time starts from zero
            # data["Press"],
            uniform_filter1d(data["Press"], size=int(5e2)),  # rolling average
            label=LOOP_VAR_NAME + "= " + str(int(total_loop_var)),
        )

        end_index = int(len(data["Press"]))
        start_index = int(0.9 * end_index)
        # selecting only final 10% of data to average, so sys has reached equilibrium

        final_values[0, j] = loop_var_values[j]
        final_values[1, j] = np.mean(data["Press"][start_index:end_index])
        final_values[2, j] = np.mean(data["PotEng"][start_index:end_index])
        final_values[3, j] = np.mean(data["Volume"][start_index:end_index])

    vol_frac_data = vol_frac(final_values[3, :])
    print("Volume Fractions: " + str(vol_frac_data))
    print(data.dtype.names)

    plt.xlabel("Time Step")
    plt.ylabel("Pressure (Natural Units)")
    plt.title("Evolution of Thermodynamic Variables at different " + LOOP_VAR_NAME)
    plt.legend()
    plt.savefig("pressureplot_frac.png")
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

