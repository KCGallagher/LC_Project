#! /usr/bin/python

""" Written by: Iria Pantazi 
    last updated: 2019/09/20 
    Instructions: (soon)
    Requirements: (soon)

    Updated to account for variable particle mol_length -KG

    Adapted significantly by KG to produce crystalline array. takes 3 input arguments (x_num, y_num, z_num), 
    as well as flag for mode (-g for generator). I.e. py nunchuck.py -g 16 4 16
"""

import time
import argparse

parser = argparse.ArgumentParser(description="generate nunchunks of 10 beads")
parser.add_argument(
    "-g",
    "--generate",
    nargs="+",
    default=None,
    help="generate new configuration from scratch.",
)
args = parser.parse_args()
import math
import numpy as np
from numpy import linalg as LA
from random import random


# unchanged parameters for the beads:
mass = 1.0
dist = 0.98  # by default
rad = 0.56
mol_length = (9 * dist) + (2 * rad)
# ie for 9 bonds, plus two ends of the molecule


def plot_all(accepted, n_mol, box_lim):
    import matplotlib.pyplot as plt

    # import itertools
    import pylab

    # from   itertools import product,imap
    # from   matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pylab as pylab

    params = {
        "legend.fontsize": "large",
        "figure.figsize": (11, 6),
        "axes.labelsize": "x-large",
        "axes.titlesize": "x-large",
        "xtick.labelsize": "x-large",
        "ytick.labelsize": "x-large",
    }
    pylab.rcParams.update(params)
    import matplotlib.cm as cm
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    from mpl_toolkits.mplot3d.proj3d import proj_transform
    from matplotlib.text import Annotation
    from mpl_toolkits.mplot3d import Axes3D

    colour = []
    x_list = [row[0] for row in accepted]
    y_list = [row[1] for row in accepted]
    z_list = [row[2] for row in accepted]
    cols = cm.seismic(np.linspace(0, 10, 10 * n_mol) / 10)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x_list, y_list, z_list, c=cols, marker="o", s=350)
    ax.set_xlim(-box_lim, box_lim)
    ax.set_ylim(
        -elongation * box_lim, elongation * box_lim
    )  # change this to get oblong box
    ax.set_zlim(-box_lim, box_lim)
    ax.grid(False)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    plt.show()
    return ()


def print_formatted_file(acc, n_molecules, mass):
    from shutil import copyfile

    with open("input_data.file", "w") as g:
        g.write("LAMMPS nunchunk data file \n\n")
        atoms = 10 * n_molecules
        bonds = 9 * n_molecules
        angles = 8 * n_molecules
        dihedrals = 0 * n_molecules
        impropers = 0 * n_molecules
        g.write("%d  atoms          \n" % atoms)
        g.write("%d  bonds          \n" % bonds)
        g.write("%d  angles         \n" % angles)
        g.write("%d  dihedrals      \n" % dihedrals)
        g.write("%d  impropers    \n\n" % impropers)

        g.write("10 atom types     \n")
        g.write("9  bond types     \n")
        g.write("8  angle types    \n")
        g.write("0  dihedral types \n")
        g.write("0  improper types \n\n")

        g.write("-%f %f xlo xhi  \n" % (0, x_num * (1 + spacer)))
        g.write("-%f %f ylo yhi  \n" % (0, y_num * (mol_length + spacer)))
        g.write("-%f %f zlo zhi  \n\n" % (0, z_num * (1 + spacer)))

        g.write("Masses\n\n")
        g.write("\t 1  %s \n" % mass)
        g.write("\t 2  %s \n" % mass)
        g.write("\t 3  %s \n" % mass)
        g.write("\t 4  %s \n" % mass)
        g.write("\t 5  %s \n" % mass)
        g.write("\t 6  %s \n" % mass)
        g.write("\t 7  %s \n" % mass)
        g.write("\t 8  %s \n" % mass)
        g.write("\t 9  %s \n" % mass)
        g.write("\t 10 %s \n\n" % mass)

        g.write("Atoms \n\n")
        for i in range(0, n_molecules, 1):

            # N molecule-tag atom-type q x y z nx ny nz
            g.write(
                "\t %d %d %d %s %s %s %d %d %d \n"
                % (
                    10 * i + 1,
                    i + 1,
                    1,
                    acc[10 * i][0],
                    acc[10 * i][1],
                    acc[10 * i][2],
                    0,
                    0,
                    0,
                )
            )
            g.write(
                "\t %d %d %d %s %s %s %d %d %d \n"
                % (
                    10 * i + 2,
                    i + 1,
                    2,
                    acc[10 * i + 1][0],
                    acc[10 * i + 1][1],
                    acc[10 * i + 1][2],
                    0,
                    0,
                    0,
                )
            )
            g.write(
                "\t %d %d %d %s %s %s %d %d %d \n"
                % (
                    10 * i + 3,
                    i + 1,
                    3,
                    acc[10 * i + 2][0],
                    acc[10 * i + 2][1],
                    acc[10 * i + 2][2],
                    0,
                    0,
                    0,
                )
            )
            g.write(
                "\t %d %d %d %s %s %s %d %d %d \n"
                % (
                    10 * i + 4,
                    i + 1,
                    4,
                    acc[10 * i + 3][0],
                    acc[10 * i + 3][1],
                    acc[10 * i + 3][2],
                    0,
                    0,
                    0,
                )
            )
            g.write(
                "\t %d %d %d %s %s %s %d %d %d \n"
                % (
                    10 * i + 5,
                    i + 1,
                    5,
                    acc[10 * i + 4][0],
                    acc[10 * i + 4][1],
                    acc[10 * i + 4][2],
                    0,
                    0,
                    0,
                )
            )
            g.write(
                "\t %d %d %d %s %s %s %d %d %d \n"
                % (
                    10 * i + 6,
                    i + 1,
                    6,
                    acc[10 * i + 5][0],
                    acc[10 * i + 5][1],
                    acc[10 * i + 5][2],
                    0,
                    0,
                    0,
                )
            )
            g.write(
                "\t %d %d %d %s %s %s %d %d %d \n"
                % (
                    10 * i + 7,
                    i + 1,
                    7,
                    acc[10 * i + 6][0],
                    acc[10 * i + 6][1],
                    acc[10 * i + 6][2],
                    0,
                    0,
                    0,
                )
            )
            g.write(
                "\t %d %d %d %s %s %s %d %d %d \n"
                % (
                    10 * i + 8,
                    i + 1,
                    8,
                    acc[10 * i + 7][0],
                    acc[10 * i + 7][1],
                    acc[10 * i + 7][2],
                    0,
                    0,
                    0,
                )
            )
            g.write(
                "\t %d %d %d %s %s %s %d %d %d \n"
                % (
                    10 * i + 9,
                    i + 1,
                    9,
                    acc[10 * i + 8][0],
                    acc[10 * i + 8][1],
                    acc[10 * i + 8][2],
                    0,
                    0,
                    0,
                )
            )
            g.write(
                "\t %d %d %d %s %s %s %d %d %d \n"
                % (
                    10 * i + 10,
                    i + 1,
                    10,
                    acc[10 * i + 9][0],
                    acc[10 * i + 9][1],
                    acc[10 * i + 9][2],
                    0,
                    0,
                    0,
                )
            )

        g.write("\n\n")
        g.write("Bonds \n\n")
        for i in range(0, n_molecules, 1):
            # N bond-type atom1-atom2
            g.write("\t %d %d %d %d \n" % (9 * i + 1, 1, 10 * i + 1, 10 * i + 2))
            g.write("\t %d %d %d %d \n" % (9 * i + 2, 2, 10 * i + 2, 10 * i + 3))
            g.write("\t %d %d %d %d \n" % (9 * i + 3, 3, 10 * i + 3, 10 * i + 4))
            g.write("\t %d %d %d %d \n" % (9 * i + 4, 4, 10 * i + 4, 10 * i + 5))
            g.write("\t %d %d %d %d \n" % (9 * i + 5, 5, 10 * i + 5, 10 * i + 6))
            g.write("\t %d %d %d %d \n" % (9 * i + 6, 6, 10 * i + 6, 10 * i + 7))
            g.write("\t %d %d %d %d \n" % (9 * i + 7, 7, 10 * i + 7, 10 * i + 8))
            g.write("\t %d %d %d %d \n" % (9 * i + 8, 8, 10 * i + 8, 10 * i + 9))
            g.write("\t %d %d %d %d \n" % (9 * i + 9, 9, 10 * i + 9, 10 * i + 10))

        g.write("\n\n")
        g.write("Angles \n\n")
        for i in range(0, n_molecules, 1):
            # N angle-type atom1-atom2(central)-atom3
            g.write(
                "\t %d %d %d %d %d \n"
                % (8 * i + 1, 1, 10 * i + 1, 10 * i + 2, 10 * i + 3)
            )
            g.write(
                "\t %d %d %d %d %d \n"
                % (8 * i + 2, 2, 10 * i + 2, 10 * i + 3, 10 * i + 4)
            )
            g.write(
                "\t %d %d %d %d %d \n"
                % (8 * i + 3, 3, 10 * i + 3, 10 * i + 4, 10 * i + 5)
            )
            g.write(
                "\t %d %d %d %d %d \n"
                % (8 * i + 4, 4, 10 * i + 4, 10 * i + 5, 10 * i + 6)
            )
            g.write(
                "\t %d %d %d %d %d \n"
                % (8 * i + 5, 5, 10 * i + 5, 10 * i + 6, 10 * i + 7)
            )
            g.write(
                "\t %d %d %d %d %d \n"
                % (8 * i + 6, 6, 10 * i + 6, 10 * i + 7, 10 * i + 8)
            )
            g.write(
                "\t %d %d %d %d %d \n"
                % (8 * i + 7, 7, 10 * i + 7, 10 * i + 8, 10 * i + 9)
            )
            g.write(
                "\t %d %d %d %d %d \n"
                % (8 * i + 8, 8, 10 * i + 8, 10 * i + 9, 10 * i + 10)
            )

    return ()


# -----------------------------------------------------------


if args.generate:  # ie argument -g

    import numpy as np
    from shutil import copyfile

    # inititalisation
    x_num = int(args.generate[0])
    y_num = int(args.generate[1])
    z_num = int(args.generate[2])
    n_molecules = x_num * y_num * z_num
    spacer = 2 * rad - dist

    accpt_mol = np.zeros((10 * n_molecules, 3))
    shuffle_molecules = False

    start_time = time.time()
    for n in range(n_molecules):
        mol_pos_index = [
            (n // y_num) % x_num,
            n % y_num,
            n // (x_num * y_num),
        ]
        # print(mol_pos_index)
        for i in range(10):
            # iterate over atoms in each molecule
            accpt_mol[10 * n + i, :] = [
                mol_pos_index[0] * (1 + spacer),
                mol_pos_index[1] * (mol_length + spacer) + i * dist,
                mol_pos_index[2] * (1 + spacer),
            ]
            # aligns all molecules along the long y axis
    accpt_mol = accpt_mol + rad  # constant offset

    if shuffle_molecules:
        rng = np.random.default_rng()
        print(accpt_mol)
        rng.shuffle(accpt_mol, axis=0)
        print(accpt_mol)

    end_time = time.time()
    print("\n time for execution: " + str(end_time - start_time) + " seconds \n")

    # ---------------------------plot all------------------------------
    # plot_all(accpt_mol, n_molecules, box_limit)

    # --------------------------print all--------------------------------
    print_formatted_file(accpt_mol, n_molecules, mass)
    print("done")

    # ---------------- rename the input_data.file ------------------------
    box_volume = (
        (x_num * (1 + spacer))
        * (y_num * (mol_length + spacer))
        * (z_num * (1 + spacer))
    )  # for inclusion in filename if desired
    src = "input_data.file"
    dst = "input_data_nunchucks_" + str(n_molecules) + ".file"
    copyfile(src, dst)

"""Frankly there is no need for the replotting option; OVITO recognises the LAMMPS input file format,
 if you specify the atom_style as molecular. It gives a much better visualisation of the particles."""
