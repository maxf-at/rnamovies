#!/usr/bin/env python3
# coding: utf-8

import forgi # only for residue coloring
import matplotlib.patches as mpatches
import sys
import random
import string
import RNA
import numpy as np
import os
import subprocess
from dataclasses import dataclass, field

import pandas as pd
# import seaborn as sns
from scipy import optimize

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, CircleCollection, EllipseCollection
import matplotlib.patheffects as path_effects
import matplotlib.path as mpath
Path = mpath.Path




@dataclass
class pos_data:
    backbone: list
    aucg_bonds: list
    gu_bonds: list
    annotation_lines: list
    annotation_chars: list
    annotation_numbers: list


@dataclass
class circular_data:
    textpos: list
    backbone: list
    lines: list
    annotation_lines: list
    annotation_numbers: list


def matrix_rotation(p, origin=(0, 0), degrees=0):
    # cite source
    if not origin:
        origin = p.mean(axis=0)
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)


class pyrnaplot:

    def __init__(self, sequence, s) -> None:

        self.sequence = sequence
        self.s = s
        self.pt = RNA.ptable(self.s)

        self.textoffset = 0, 0
        self.plot_scalar = 0

        self.coords_naview = []
        self.naview_layout = None  # pos_data dataclass

        self.coords_circular = []
        self.circular_layout = None  # pos_data dataclass

        # colouring
        cg = forgi.load_rna(self.s, "any", allow_many=True)[0]
        self.cstring = cg.to_element_string()
        self.forgi_coding = {'f': 'orange',
                             't': 'orange',
                             's': 'green',
                             'h': 'blue',
                             'i': 'yellow',
                             'm': 'red'}

    def get_naview_coords(self, rotate=0):
        c = []
        for i, coord in enumerate(RNA.naview_xy_coordinates(self.s)):
            if i == len(self.s):
                break
            c.append((coord.X, coord.Y))
        c = np.array(c).astype(float)
        if rotate != 0:
            c = matrix_rotation(c, origin=False, degrees=-rotate)
        return c

    def get_circular_coords(self):
        # this just creates dot along a unit circle with len(s) points
        c = []
        for i, coord in enumerate(RNA.simple_circplot_coordinates(self.s)):
            if i == len(self.s):
                break
            c.append((coord.X, coord.Y))
        c = np.array(c).astype(float)
        return c

    def naview_plot_layout(self, coordinates=False):

        if not isinstance(coordinates, bool):
            self.coords_naview = coordinates
        else:
            self.coords_naview = self.get_naview_coords()

        c = self.coords_naview

        backbone = []
        aucg_bonds = []
        gu_bonds = []
        annotation_lines = []
        annotation_chars = []
        annotation_numbers = []

        for i, ch in enumerate(self.sequence):

            x, y = c[i] + self.textoffset

            # print (i, c, x, y)
            annotation_chars.append(((x, y), ch))
            # ax.text(x, y, ch, size=16*plot_modifier,  ha="center", va="center")

            # AU, CG, GU bonds
            if self.pt[i+1] > i:
                # print ("link", i, pt1[i+1]-1)
                char1 = self.sequence[i]
                char2 = self.sequence[self.pt[i+1]-1]

                pos1 = c[i]
                pos2 = c[self.pt[i+1]-1]
                if pos1[0] < pos2[0]:
                    pos1, pos2 = pos2, pos1

                bond_identifier = str(i+1) + "-" + str(self.pt[i+1])

                if (char1 == "G" and char2 == "U") or (char1 == "U" and char2 == "G"):
                    midpoint = (pos1+pos2)/2
                    # gu_bonds.append((midpoint[0], midpoint[1]))
                    gu_bonds.append((bond_identifier, midpoint))

                else:
                    v = pos2-pos1
                    l = np.linalg.norm(v)  # euclidean distance
                    v /= l
                    pos1 = pos1 + v*5
                    pos2 = pos2 - v*5
                    aucg_bonds.append((bond_identifier, pos1, pos2))

            # backbone
            if i+1 == len(self.sequence):
                continue

            pos1 = c[i]
            pos2 = c[i+1]
            if pos1[0] < pos2[0]:
                pos1, pos2 = pos2, pos1

            v = pos2-pos1

            l = np.linalg.norm(v)
            if l > 12:  # ignore short lines
                v /= l  # normalize to unit vector

                cutoff = 1.2*self.plot_scalar
                pos1 = pos1 + v*6  # 4
                pos2 = pos2 - v*6
                link = [pos1, pos2]
                # line1, = ax.plot(*zip(*link))
                # links.append(((pos1[0], pos1[1]),(pos2[0], pos2[1])))
                backbone.append((pos1, pos2))
            else:
                # add a dummy placeholder backbone if invisible
                backbone.append(((pos1+pos2)/2, (pos1+pos2)/2))

            # annotations: calculate normal vector
            if i % 10 == 9 or i == 0:  # start counting at 1, not 0

                if i == 0:
                    lastpos = c[i]
                else:
                    lastpos = c[i-1]

                nextpos = c[i+1]
                v = nextpos-lastpos

                l = np.linalg.norm(v)
                v /= l  # normalize to unit vector
                v = np.array([v[1], -v[0]])  # 90° rotation

                pos1 = c[i] + v*6
                pos2 = pos1 + v*18  # 13 length vector
                charpos = pos2 + v*6

                annotation_lines.append((pos1, pos2))
                annotation_numbers.append((charpos, i+1))

        self.naview_layout = pos_data(
            backbone, aucg_bonds, gu_bonds, annotation_lines, annotation_chars, annotation_numbers)
        return self.naview_layout

    def naview_plot(self, dpi=72, rotate=0, coordinates=False):

        textoffset = -0.5, -0.9
        # textoffset = 0, 0

        border = 30

        if not isinstance(coordinates, bool):
            self.coords_naview = coordinates
        else:
            self.coords_naview = self.get_naview_coords()

        if rotate != 0:
            self.coords_naview = matrix_rotation(
                self.coords_naview, origin=False, degrees=-rotate)

        self.naview_plot_layout(coordinates)
        c = self.coords_naview

        # set plot dimensions and estimate a scalar for text / object rendering
        limits = c.min()*1.1, c.max()*1.1
        xmax, ymax = c.max(axis=0) + [border, border]
        xmin, ymin = c.min(axis=0) - [border, border]
        xlength = abs(xmax-xmin)
        ylength = abs(ymax-ymin)
        ratio = xlength/ylength
        # inverse distance, adjust font size / bond size according to total plot length
        self.plot_scalar = 1/abs(limits[1]-limits[0])*300
        if ratio > 1:
            self.plot_scalar *= ratio

        # matplotlib start

        fig = plt.figure(figsize=(10*ratio, 10), dpi=dpi)
        ax = fig.add_subplot(1, 1, 1)

        # plt.rcParams.update({
        #     "text.usetex": False,
        #     # "font.family": "serif",
        #     "font.family": "sans-serif",
        #     "font.sans-serif": ["Helvetica"]})

        plt.tight_layout()

        ax.set_xlim((xmin, xmax))
        ax.set_ylim((ymin, ymax))
        ax.axis("off")
        ax.scatter(c[:, 0], c[:, 1], alpha=0.0)

        # get the location of all our text & lines
        l = self.naview_layout

        # discard info which would interfer with LineCollection (identifiers i[0] are needed for movies)
        l.aucg_bonds = [(i[1], i[2]) for i in l.aucg_bonds]
        l.gu_bonds = [i[1] for i in l.aucg_bonds]

        backbonecollection = LineCollection(
            l.backbone, colors="grey", linewidths=2.0*self.plot_scalar, path_effects=[path_effects.Stroke(capstyle="round")])
        aucgcollection = LineCollection(l.aucg_bonds, colors="red", linewidths=2.5 *
                                        self.plot_scalar, path_effects=[path_effects.Stroke(capstyle="round")])
        annotationcollection = LineCollection(
            l.annotation_lines, colors="grey", linewidths=1.2*self.plot_scalar, path_effects=[path_effects.Stroke(capstyle="round")])

        ax.add_collection(backbonecollection)
        ax.add_collection(aucgcollection)
        ax.add_collection(annotationcollection)

        circlesizes = [8*self.plot_scalar**2] * len(l.gu_bonds)
        coll = CircleCollection(
            circlesizes, offsets=l.gu_bonds, color="red", transOffset=ax.transData)
        ax.add_collection(coll)

        for i, ((x, y), ch) in enumerate(l.annotation_chars):
            text = ax.text(x, y, ch, size=16*self.plot_scalar,
                           ha="center", va="center")
            c = self.forgi_coding[self.cstring[i]]
            text.set_path_effects([path_effects.Stroke(linewidth=4, foreground=c, alpha=0.3),
                                   path_effects.Normal()])

        for (x, y), ch in l.annotation_numbers:
            ax.text(x, y, ch, size=12*self.plot_scalar,
                    ha="center", va="center")

    def circular_plot_layout(self):
        if not self.coords_circular:
            self.coords_circular = self.get_circular_coords()

        r = self.coords_circular

        angle_multiplier = 4
        line_modifier = 1/len(self.sequence) * 100

        # print (line_modifier)
        textpos = []
        backbone = []
        lines = []
        annotation_lines = []
        annotation_numbers = []

        for i, e in enumerate(self.sequence):
            pos = r[i]*1.08
            textpos.append((pos, e))

            # circle bezier segments
            if i+1 != len(self.sequence):
                pos1 = r[i]
                pos2 = r[i+1]
                midpoint = (pos2+pos1)/2
                midpoint /= np.linalg.norm(midpoint)  # * 0.99
                v = midpoint-pos1
                v += pos1
                backbone.append((pos1, v, pos2))

            # annotations: calculate normal vector
            if i % 10 == 9 or i == 0:  # start counting at 1, not 0
                pos1 = r[i]
                pos2 = r[i]*1.14
                pos3 = r[i]*1.18
                annotation_lines.append((pos1, pos2))
                annotation_numbers.append((pos3, i+1))

            if self.pt[i+1] > i:
                # print ("link", i, pt1[i+1]-1)
                char1 = self.sequence[i]
                char2 = self.sequence[self.pt[i+1]-1]

                # position 1 and adjacent nodes for normal vector
                if i == 0:
                    pos1l = r[-1]
                else:
                    pos1l = r[i-1]
                pos1 = r[i]
                pos1r = r[i+1]

                pos2l = r[self.pt[i+1]-2]
                pos2 = r[self.pt[i+1]-1]
                if self.pt[i+1] == len(self.sequence):
                    pos2r = r[0]
                else:
                    pos2r = r[self.pt[i+1]]

                midpoint = (pos1+pos2)/2
                midpointdist = np.linalg.norm(midpoint)  # distance to origin

                l12 = np.linalg.norm(pos2-pos1) / angle_multiplier

                # print ("l", l12, midpointdist)

                v1 = pos1l-pos1r
                v1 = np.array([v1[1], -v1[0]])  # 90° rotation
                v1 /= np.linalg.norm(v1)
                v1 *= l12

                bezierpos1 = pos1+v1

                v2 = pos2l-pos2r
                v2 = np.array([v2[1], -v2[0]])  # *15  # 90° rotation
                v2 /= np.linalg.norm(v2)
                v2 *= l12

                bezierpos2 = pos2+v2

                vmid = pos2-pos1
                vmid = np.array([vmid[1], -vmid[0]]) * \
                    midpointdist / angle_multiplier

                # always move towards the centre - flip nodes if necessary
                if np.linalg.norm(midpoint + vmid) < midpointdist:
                    beziermidpos = midpoint + vmid
                else:
                    beziermidpos = midpoint - vmid

                # # bezier curves
                lines.append(
                    (pos1, bezierpos1, beziermidpos, bezierpos2, pos2))

        self.circular_layout = circular_data(
            textpos, backbone, lines, annotation_lines, annotation_numbers)
        return self.circular_layout

    def circular_plot(self, dpi=72):

        border = 30
        angle_multiplier = 4
        line_modifier = 1/len(self.sequence) * 100
        xy_limit = 1.2

        # if self.coords_circular == None:
        self.circular_plot_layout()
        r = self.coords_circular

        # matplotlib start

        fig = plt.figure(figsize=(10, 10), dpi=dpi)
        ax = fig.add_subplot(1, 1, 1)

        # plt.rcParams.update({
        #     "text.usetex": False,
        #     # "font.family": "serif",
        #     "font.family": "sans-serif",
        #     "font.sans-serif": ["Helvetica"]})

        plt.tight_layout()

        ax.set_xlim((-xy_limit, xy_limit))
        ax.set_ylim((-xy_limit, xy_limit))
        ax.axis("off")

        # get the location of all our text & lines
        l = self.circular_layout

        # Annotations
        for i, ((x, y), ch) in enumerate(l.textpos):
            text = ax.text(x, y, ch, size=12*line_modifier,
                           ha="center", va="center")
            c = self.forgi_coding[self.cstring[i]]
            text.set_path_effects([path_effects.Stroke(linewidth=4, foreground=c, alpha=0.3),
                                   path_effects.Normal()])

        annotationcollection = LineCollection(l.annotation_lines, colors="grey", linewidths=1 *
                                              line_modifier, alpha=0.4, path_effects=[path_effects.Stroke(capstyle="round")])
        ax.add_collection(annotationcollection)

        for (x, y), ch in l.annotation_numbers:
            ax.text(x, y, ch, size=8*line_modifier,
                    ha="center", va="center", color="black")

        # render bezier lines
        for a, b, c in l.backbone:
            curve = mpatches.PathPatch(
                Path([a, b, c],
                     [Path.MOVETO, Path.CURVE3, Path.CURVE3]),
                fc="none", transform=ax.transData, color="green")
            ax.add_patch(curve)

        for a, b, c, d, e in l.lines:
            curve = mpatches.PathPatch(
                Path([a, b, c, d, e],
                     [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.CURVE3, Path.CURVE3]),
                fc="none", transform=ax.transData, color="red", linewidth=1*line_modifier, linestyle="dashed")  # densely dashed  - dashdot
            ax.add_patch(curve)

        # dummy line - otherwise the last line is thicker ?!?
        dummy = mpatches.PathPatch(
            Path([(0, 0)], [Path.MOVETO]), fc="none", transform=ax.transData)
        ax.add_patch(dummy)

        return plt
