#! /usr/bin/python

""" Written by KG to produce crystalline array of molecules
    Based on code for dilute configurations by Iria Pantazi

    Takes 4 input arguments (mol_length, x_num, y_num, z_num), 
    as well as flag for mode (-g for generator). I.e. py nunchuck.py -g 15 16 4 16
    Note that i_num is the number of molecules in the i-th axis
    This script automically aligns molecules along the y axis
"""

import time
import argparse

parser = argparse.ArgumentParser(description="generate nunchunks of 'mol_length' beads")
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
    cols = cm.seismic(np.linspace(0, mol_length, mol_length * n_mol) / mol_length)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x_list, y_list, z_list, c=cols, marker="o", s=350)
    ax.set_xlim(0, x_num * (1 + spacer))
    ax.set_ylim(0, y_num * (end_to_end_length + spacer))
    ax.set_zlim(0, z_num * (1 + spacer))
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
        atoms = mol_length * n_molecules
        bonds = (mol_length - 1) * n_molecules
        angles = (mol_length - 2) * n_molecules
        dihedrals = 0 * n_molecules
        impropers = 0 * n_molecules
        g.write("%d  atoms          \n" % atoms)
        g.write("%d  bonds          \n" % bonds)
        g.write("%d  angles         \n" % angles)
        g.write("%d  dihedrals      \n" % dihedrals)
        g.write("%d  impropers    \n\n" % impropers)

        g.write("%d  atom types          \n" % mol_length)
        g.write("%d  bond types         \n" % (mol_length - 1))
        g.write("%d  angle types         \n" % (mol_length - 2))
        g.write("0  dihedral types \n")
        g.write("0  improper types \n\n")

        g.write("%f %f xlo xhi  \n" % (0, x_num * (1 + spacer)))
        g.write("%f %f ylo yhi  \n" % (0, y_num * (end_to_end_length + spacer)))
        g.write("%f %f zlo zhi  \n\n" % (0, z_num * (1 + spacer)))

        g.write("Masses\n\n")
        for i in range(mol_length):
            g.write("\t %d  %s \n" % ((i + 1), mass))

        g.write("\nAtoms \n\n")
        for i in range(0, n_molecules, 1):
            for j in range(mol_length):
                # N molecule-tag atom-type q x y z nx ny nz
                g.write(
                    "\t %d %d %d %s %s %s %d %d %d \n"
                    % (
                        mol_length * i + 1 + j,
                        i + 1,
                        1 + j,
                        acc[mol_length * i + j][0],
                        acc[mol_length * i + j][1],
                        acc[mol_length * i + j][2],
                        0,
                        0,
                        0,
                    )
                )

        g.write("\n\n")
        g.write("Bonds \n\n")
        for i in range(0, n_molecules, 1):
            for j in range(mol_length - 1):
                # N bond-type atom1-atom2
                g.write(
                    "\t %d %d %d %d \n"
                    % (
                        (mol_length - 1) * i + 1 + j,
                        1 + j,
                        mol_length * i + 1 + j,
                        mol_length * i + 2 + j,
                    )
                )

        g.write("\n\n")
        g.write("Angles \n\n")
        for i in range(0, n_molecules, 1):
            for j in range(mol_length - 2):
                # N angle-type atom1-atom2(central)-atom3
                g.write(
                    "\t %d %d %d %d %d \n"
                    % (
                        (mol_length - 2) * i + 1 + j,
                        1 + j,
                        mol_length * i + 1 + j,
                        mol_length * i + 2 + j,
                        mol_length * i + 3 + j,
                    )
                )

    return ()


# -----------------------------------------------------------


if args.generate:  # ie argument -g

    import numpy as np
    from shutil import copyfile

    # inititalisation
    mol_length = int(args.generate[0])
    x_num = int(args.generate[1])
    y_num = int(args.generate[2])
    z_num = int(args.generate[3])
    n_molecules = x_num * y_num * z_num
    spacer = 2 * rad - dist

    end_to_end_length = ((mol_length - 1) * dist) + (2 * rad)
    # ie for 9 bonds, plus two ends of the molecule

    accpt_mol = np.zeros((mol_length * n_molecules, 3))
    shuffle_molecules = False

    start_time = time.time()
    for n in range(n_molecules):
        mol_pos_index = [
            (n // y_num) % x_num,
            n % y_num,
            n // (x_num * y_num),
        ]
        # print(mol_pos_index)
        for i in range(mol_length):
            # iterate over atoms in each molecule
            accpt_mol[mol_length * n + i, :] = [
                mol_pos_index[0] * (1 + spacer),
                mol_pos_index[1] * (end_to_end_length + spacer) + i * dist,
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
        * (y_num * (end_to_end_length + spacer))
        * (z_num * (1 + spacer))
    )  # for inclusion in filename if desired
    src = "input_data.file"
    dst = "input_data_nunchucks_" + str(n_molecules) + "_" + str(mol_length) + ".file"
    copyfile(src, dst)

"""Frankly there is no need for the replotting option; OVITO recognises the LAMMPS input file format,
 if you specify the atom_style as molecular. It gives a much better visualisation of the particles."""
