#! /usr/bin/python

""" Written by: Iria Pantazi 
    last updated: 2019/09/20 
    Instructions: (soon)
    Requirements: (soon)
"""

import time
import argparse

parser = argparse.ArgumentParser(description="generate nunchunks of 10 beads")
parser.add_argument(
    "-g",
    "--generate",
    nargs="+",
    default=None,
    help="generate new configuration from scratsch.",
)
parser.add_argument(
    "-r",
    "--replot",
    nargs="+",
    help="Replots what has been written in the rawdata files.",
)
parser.add_argument("-a", "--analyse", nargs="+", help="to be included shortly")
parser.add_argument("-d", "--rdf", nargs="+", help="to be included shortly")
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


def within_box(ghost, box_lim):
    flag = True
    toBreak = False
    for i in range(0, 10, 9):  # check only the edge beads
        for j in range(3):
            print(i, j)
            if abs(ghost[i][j]) > box_lim:
                print("out")
                flag = False
                toBreak = True
                break
        if toBreak == True:
            break
    return flag


def perform_rand_rot(ghost):
    new_ghost = np.zeros((10, 3))
    ran_rot = Quaternion.random()  # K changes Q to q
    for i in range(10):
        new_ghost[i] = ran_rot.rotate(ghost[i])
    return new_ghost


def they_overlap(ghost, accepted, mol, rad):
    overlap = False
    lista = list(range(10))  # [0,1,2,4,5,7,8]
    break_flag = False

    for k in range(mol):
        for i in lista:
            for j in lista:
                dist = np.linalg.norm(accepted[10 * k + i] - ghost[j])
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
    ghost = np.zeros((10, 3))
    # center 0
    ghost[0][0] = 0
    ghost[0][1] = 0
    ghost[0][2] = 0
    # tail of the next 9 beads
    for i in range(1, 10, 1):
        ghost[i] = ghost[0]
        ghost[i][2] += i * dist
    # preform random rotation
    ran_rot = Quaternion.random()  # K changes Q to q
    for i in range(10):
        ghost[i] = ran_rot.rotate(ghost[i])
    # preform random displacement
    ran_dis_x = (random() - 0.5e0) * 2.0 * (box_limit)
    ran_dis_y = (random() - 0.5e0) * 2.0 * (box_limit)
    ran_dis_z = (random() - 0.5e0) * 2.0 * (box_limit)
    for i in range(10):
        ghost[i][0] += ran_dis_x
        ghost[i][1] += ran_dis_y
        ghost[i][2] += ran_dis_z
    return ghost


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
    ax.set_ylim(-box_lim, box_lim)  # change this to get oblong box
    ax.set_zlim(-box_lim, box_lim)
    ax.grid(False)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    plt.show()
    return ()


def print_formatted_file(acc, n_molecules, box_limit, mass):
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

        g.write("-%f %f xlo xhi  \n" % (box_limit, box_limit))
        g.write("-%f %f ylo yhi  \n" % (box_limit, box_limit))
        g.write("-%f %f zlo zhi  \n\n" % (box_limit, box_limit))

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


if args.generate:

    import numpy as np
    from shutil import copyfile

    # inititalsation
    n_molecules = int(args.generate[0])
    box_limit = float(args.generate[1]) / 2.0
    rot_threshold = 500
    ghost_mol = np.zeros((10, 3))
    accpt_mol = np.zeros((10 * n_molecules, 3))

    start_time = time.time()
    # first one is always accepted
    ghost_mol = gen_ghost(box_limit, dist)
    while within_box(ghost_mol, box_limit) == False:
        ghost_mol = gen_ghost(box_limit, dist)
    for i in range(10):
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
            flag = they_overlap(ghost_mol, accpt_mol, mol, 0.56)
        # -------if not then add them to the list of accepted----------------
        if flag == False:
            for i in range(10):
                accpt_mol[10 * mol + i] = ghost_mol[i]
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
        + str((box_limit) * 2)
        + ".file"
    )
    copyfile(src, dst)

if args.replot:
    n_molecules, box_limit = args.replot
    n_molecules = int(n_molecules)
    box_limit = float(box_limit)
    infiles = "rawdata_" + str(n_molecules) + "_" + str((box_limit) * 2)

    # ------------------- initialisation ---------------------
    accepted = np.zeros((10 * n_molecules, 3))
    i = 0

    # ------------------ call from functions.py --------------
    with open(infile, "r") as g:
        for row in g.readlines():
            accepted[i][0], accepted[i][1], accepted[i][2] = row.split()
            i += 1
        plot_all(accepted, n_molecules, box_limit)

