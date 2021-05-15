#! /usr/bin/python

""" Written by: Iria Pantazi 
    Updated by candidate to account for variable particle mol_length

    Accepts argument for mode (-g) and values for molecule number, length, box size
    Ie it can be run from shell via the command 'py nunchuck.py -g 2 10 20
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
parser.add_argument(
    "-r",
    "--replot",
    nargs="+",
    help="Replots what has been written in the rawdata files.",
)
args = parser.parse_args()
import math
import numpy as np
from numpy import linalg as LA
from random import random

# import quaternion
from pyquaternion import Quaternion

# unchanged parameters for the beads:
mass = 1.0
dist = 0.98
rad = 0.56

elongation = 1
# aspect ratio of oblong base; y axis is E times longer than x and z axes


def scale_vector(j):
    """Returns scale factor for axis of index j"""
    if j == 1:
        return elongation  # extended y axis
    elif j == 0 or j == 2:
        return 1
    else:
        raise ValueError("Unexpected Index: recieved value " + str(j))


def within_box(ghost, box_lim):
    """Returns boolean to determine whether candidate particle lies within box"""
    flag = True
    toBreak = False
    for i in range(0, mol_length, mol_length - 1):  # check only the edge beads
        for j in range(3):
            if abs(ghost[i][j]) > scale_vector(j) * box_lim:
                flag = False
                toBreak = True
                break
        if toBreak == True:
            break
    return flag


def perform_rand_rot(ghost):
    new_ghost = np.zeros((mol_length, 3))
    ran_rot = Quaternion.random()
    for i in range(mol_length):
        new_ghost[i] = ran_rot.rotate(ghost[i])
    return new_ghost


def they_overlap(ghost, accepted, mol, rad):
    overlap = False
    lista = list(range(mol_length))
    break_flag = False

    for k in range(mol):
        for i in lista:
            for j in lista:
                dist = np.linalg.norm(accepted[mol_length * k + i] - ghost[j])
                if dist < 2 * rad:
                    break_flag = True
                    overlap = True
                    break
            if break_flag == True:
                break
        if break_flag == True:
            break
    return overlap


def gen_ghost(box_limit, dist):
    ghost = np.zeros((mol_length, 3))
    # center 0
    ghost[0][0] = 0
    ghost[0][1] = 0
    ghost[0][2] = 0
    # tail of the next (mol_length-1) beads
    for i in range(1, mol_length, 1):
        ghost[i] = ghost[0]
        ghost[i][2] += i * dist
    # preform random rotation
    ran_rot = Quaternion.random()
    for i in range(mol_length):
        ghost[i] = ran_rot.rotate(ghost[i])
    # preform random displacement
    ran_dis_x = (random() - 0.5e0) * 2.0 * (box_limit)
    ran_dis_y = (random() - 0.5e0) * 2.0 * elongation * (box_limit)
    ran_dis_z = (random() - 0.5e0) * 2.0 * (box_limit)
    for i in range(mol_length):
        ghost[i][0] += ran_dis_x
        ghost[i][1] += ran_dis_y
        ghost[i][2] += ran_dis_z
    return ghost


def plot_all(accepted, n_mol, box_lim):
    """General plotting function for plotting mode"""
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
    ax.set_xlim(-box_lim, box_lim)
    ax.set_ylim(-elongation * box_lim, elongation * box_lim)
    ax.set_zlim(-box_lim, box_lim)
    ax.grid(False)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    plt.show()
    return ()


def print_formatted_file(acc, n_molecules, box_limit, mass):
    """Generates the input file for MD simulations"""
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

        g.write("-%f %f xlo xhi  \n" % (box_limit, box_limit))
        g.write("-%f %f ylo yhi  \n" % (elongation * box_limit, elongation * box_limit))
        g.write("-%f %f zlo zhi  \n\n" % (box_limit, box_limit))

        g.write("Masses\n\n")
        for i in range(mol_length):
            g.write("\t %d  %s \n" % ((i + 1), mass))

        g.write("\nAtoms \n\n")
        for i in range(n_molecules):
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
        for i in range(n_molecules):
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

    # inititalsation
    n_molecules = int(args.generate[0])
    mol_length = int(args.generate[1])
    box_limit = float(args.generate[2]) / 2.0
    rot_threshold = 500
    ghost_mol = np.zeros((mol_length, 3))
    accpt_mol = np.zeros((mol_length * n_molecules, 3))

    start_time = time.time()
    # first one is always accepted
    ghost_mol = gen_ghost(box_limit, dist)
    while within_box(ghost_mol, box_limit) == False:
        ghost_mol = gen_ghost(box_limit, dist)
    for i in range(mol_length):
        accpt_mol[i] = ghost_mol[i]
    mol = 1
    while mol < n_molecules:
        # ----------------------generate ghost molecule--------------------
        ghost_mol = gen_ghost(box_limit, dist)
        while within_box(ghost_mol, box_limit) == False:
            ghost_mol = gen_ghost(box_limit, dist)
        # -------------check overlaping and perform a mx of 200 rots-----------
        flag = they_overlap(ghost_mol, accpt_mol, mol, rad)
        attempt_num = 0
        while flag == True:
            attempt_num += 1
            if attempt_num > rot_threshold:
                mol -= 1
                break
            ghost_mol = perform_rand_rot(ghost_mol)
            while within_box(ghost_mol, box_limit) == False:
                ghost_mol = gen_ghost(box_limit, dist)
            flag = they_overlap(ghost_mol, accpt_mol, mol, rad)
        # -------if not then add them to the list of accepted----------------
        if flag == False:
            for i in range(mol_length):
                accpt_mol[mol_length * mol + i] = ghost_mol[i]
        print(mol)
        # -----------------------move to next mol--------------------------
        mol += 1
    end_time = time.time()
    print("\n time for execution: " + str(end_time - start_time) + " seconds \n")

    # ---------------------------plot all------------------------------
    plot_all(accpt_mol, n_molecules, box_limit)

    # --------------------------print all--------------------------------
    print_formatted_file(accpt_mol, n_molecules, box_limit, mass)
    print("done")

    with open("accepted.dat", "w") as f:
        string_accpt_mol = str(accpt_mol).replace("[", "").replace("]", "")
        f.writelines(string_accpt_mol + "\n")

    # --save a copy to be read by load_plot.py----------------------------
    target_name = "rawdata_" + str(n_molecules) + "_" + str((box_limit) * 2)
    copyfile("accepted.dat", target_name)

    # ---------------- rename the input_data.file ------------------------
    src = "input_data.file"
    dst = (
        "input_data_nunchucks_"
        + str(n_molecules)
        + "_"
        + str(mol_length)
        + "_"
        + str(int((box_limit) * 2))
        + ".file"
    )
    copyfile(src, dst)

if args.replot:  # ie argument -r
    n_molecules, mol_length, box_limit = args.replot
    n_molecules = int(n_molecules)
    mol_length = int(mol_length)
    box_limit = float(box_limit)
    infiles = "rawdata_" + str(n_molecules) + "_" + str((box_limit) * 2)

    # ------------------- initialisation ---------------------
    accepted = np.zeros((mol_length * n_molecules, 3))
    i = 0

    # ------------------ call from functions.py --------------
    with open(infile, "r") as g:
        for row in g.readlines():
            accepted[i][0], accepted[i][1], accepted[i][2] = row.split()
            i += 1
        plot_all(accepted, n_molecules, box_limit)

