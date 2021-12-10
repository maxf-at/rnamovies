# execute: 
# low quality:  manim -pql rnamovie.py
# high quality: manim -pqh rnamovie.py

import pandas as pd
from scipy import spatial
import numpy as np
from xml.dom import minidom
from pandas.core.indexes import base
from scipy.spatial.distance import cdist
from scipy.optimize import dual_annealing
import logging

from manim import *

# vis.py
import vis


# config.background_color = WHITE
config["pixel_height"] = 2160
config["pixel_width"] = 3840
# config["pixel_height"] = 1080
# config["pixel_width"] = 1920

# scene offsets
scene_scalar = 1/50

# animation settings
fadeinout_seconds = 0.5
wait_seconds = 0.0
transition_seconds = 0.4
# wait_seconds = 0.8
# transition_seconds = 1.2

sequence    = "GGGCCCAUAGCUCAGUGGUAGAGUGCCUCCUUUGCAAGGAGGAUGCCCUGGGUUCGAAUCCCAGUGGGUCCA"
structures = ["(((((((((((((((.((((.....((((((.....)))))).))))))))))).........)))))))).",
              "((((((((.....((.((((.....((((((.....)))))).))))))(((.......))).)))))))).",
              "(((.....(((((((.((((.....((((((.....)))))).))))))))))).....)))..((....))",
              "((((.....((((.......)))).((((((.....))))))..)))).(((.......)))..((....))",
              "((((((...((((.......)))).((((((.....)))))).....(((((.......))))).)))))).",
              "((((((.((((((.......)))).((((((.....)))))).))..(((((.......))))).)))))).",
              "((((((((.((((.......))))...((((.....))))((....)).(((.......))).)))))))).",
              "((((((...((((.......))))((.((((.....))))((....)).(((.......))).)))))))).",
              "((((.....))))..(((.......((((((.....))))))...(((((((.......))))).))..)))",
              "((((.....))))...(((.....))).(((.....))).((((.(((((((.......))))).)))))).",
              "((((.....))))..(((((...))((((((.....))))))...(((((((.......))))).))..)))",
              "((((((...((...))((((...))((((((.....))))))...))(((((.......))))).)))))).",
              "((((((((.(((....))).((...((((((.....))))))..(((...)))))........)))))))).",
              ]

# direct path example
sequence   =  "AGCAAUUGUUGUCGCGGAUGAAUAAGUUGAUUAAAUAACGUGAUGAUCCUAUAAGUCGUUGCACAUAGACUCCGCAUCGCGAUUAGCAGAAACUAUGGUC"
structures = [".((((((((.((.(((((.......((((......))))((((((((.......))))))))........))))))).)))))).)).............",
              ".((((((((..(.(((((.......((((......))))((((((((.......))))))))........))))))..)))))).)).............",
              ".((((((((....(((((.......((((......))))((((((((.......))))))))........)))))...)))))).)).............",
              ".((.(((((....(((((.......((((......))))((((((((.......))))))))........)))))...)))))..)).............",
              ".((..((((....(((((.......((((......))))((((((((.......))))))))........)))))...))))...)).............",
              ".(...((((....(((((.......((((......))))((((((((.......))))))))........)))))...))))....).............",
              ".....((((....(((((.......((((......))))((((((((.......))))))))........)))))...))))..................",
              "......(((....(((((.......((((......))))((((((((.......))))))))........)))))...)))...................",
              ".......((....(((((.......((((......))))((((((((.......))))))))........)))))...))....................",
              ".......(.....(((((.......((((......))))((((((((.......))))))))........)))))....)....................",
              ".............(((((.......((((......))))((((((((.......))))))))........))))).........................",
              "............((((((.......((((......))))((((((((.......))))))))........))))).....)...................",
              "...........(((((((.......((((......))))((((((((.......))))))))........))))).....))..................",
              ".......(...(((((((.......((((......))))((((((((.......))))))))........))))).....))....).............",
              "......((...(((((((.......((((......))))((((((((.......))))))))........))))).....))....))............",
              "......(((..(((((((.......((((......))))((((((((.......))))))))........))))).....))...)))............",
              "......((((.(((((((.......((((......))))((((((((.......))))))))........))))).....))..))))............",
              "......((((((((((((.......((((......))))((((((((.......))))))))........))))).....)).)))))............",
              "......((((((((((((.......(((........)))((((((((.......))))))))........))))).....)).)))))............",
              "......(((((((.((((.......(((........)))((((((((.......))))))))........))))......)).)))))............",
              "......((((((((((((.......(((........)))((((((((.......))))))))........)))).....))).)))))............",
              "......((((((((.(((.......(((........)))((((((((.......))))))))........)))......))).)))))............",
              "......((((((((((((.......(((........)))((((((((.......))))))))........))).....)))).)))))............",
              "......(((((((((.((.......(((........)))((((((((.......))))))))........))......)))).)))))............",
              "......((((((((((((.......(((........)))((((((((.......))))))))........)).....))))).)))))............",
              "......(((((((((((........(((........)))((((((((.......)))))))).........).....))))).)))))............",
              "......((((((((((.........(((........)))((((((((.......))))))))...............))))).)))))............",
              "......((((((((((.........(.(........).)((((((((.......))))))))...............))))).)))))............",
              "......((((((((((...........(........)..((((((((.......))))))))...............))))).)))))............",
              "......((((((((((.......................((((((((.......))))))))...............))))).)))))............",
              ]

def mlog(*x):
    logging.getLogger("manim").info(*x)

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


def min_distances(sequence, structures, init_rotation=0, xylimit=150, rlimit=20):

    xylimit = -xylimit, xylimit

    # list of numpy 2d arrays
    coordinates = [vis.pyrnaplot(sequence, s).get_naview_coords()
                   for s in structures]

    if init_rotation != 0:
        coordinates[0] = matrix_rotation(
            coordinates[0], origin=False, degrees=init_rotation)

    def objective(bounds):
        # function for annealing optimization
        x1, y1, r1 = bounds
        c2_temp = matrix_rotation(c2, origin=False, degrees=r1) + (x1, y1)
        # c2b = c2+ (x1,y1)
        metric = "euclidean"
        # metric = "hamming"
        dist1 = cdist(c1, c2_temp, metric=metric).sum()
        return dist1

    last_rotation = init_rotation

    for i in range(0, len(structures)-1):

        current_rlimit = last_rotation-rlimit, last_rotation+rlimit
        bounds = [xylimit, xylimit, current_rlimit]

        s1 = structures[i]
        s2 = structures[i+1]

        c1 = coordinates[i]
        c2 = coordinates[i+1]

        mlog(s1)
        mlog(s2)

        # dual annealing search
        result = dual_annealing(objective, bounds, maxiter=100)

        # evaluate solution
        x1, y1, r1 = result['x']

        mlog('Status : %s' % result['message'])
        # evaluation = objective(solution)
        
        mlog(f"offset: x={x1:.2f} / y={y1:.2f}, rotation: {r1:.2f}, evaluations: {result['nfev']}")
        
        coordinates[i+1] = matrix_rotation(c2,
                                           origin=False, degrees=r1) + (x1, y1)

        mlog("-")
        last_rotation = r1

    # fig = plt.figure(figsize=(10, 10), dpi=72)
    # for s, c in zip(structures, coordinates):
    #     sns.scatterplot(x=c[:,0], y=c[:,1])
        # vis.pyrnaplot(sequence, s).naview_plot(coordinates=c, dpi=30)

    return coordinates



class rnamovie(Scene):
    def construct(self):

        coordinates = min_distances(
            sequence, structures, init_rotation=100, rlimit=30)

        # calculate an x,y offset from all coordinates
        midpoint = []
        for c in coordinates:
            midpoint.append(c.mean(axis=0))
        midpoint = np.array(midpoint).mean(axis=0)

        # adjust offset
        coordinates = [c-midpoint for c in coordinates]

        # calculate position of characters, lines, ...
        layout_data = [vis.pyrnaplot(sequence, s).naview_plot_layout(
            c) for s, c in zip(structures, coordinates)]

        backbone_links = pd.DataFrame([i.backbone for i in layout_data])
        annotation_chars = pd.DataFrame(
            [i.annotation_chars for i in layout_data])
        annotation_numbers = [i.annotation_numbers for i in layout_data]
        annotation_lines = [i.annotation_lines for i in layout_data]
        aucg_bonds = pd.DataFrame()
        gu_bonds = pd.DataFrame()

        # iterate over all bonds: every bond has an identifier (nt1-nt2)
        # only equal bonds are allowed to morph into each other during the animation,
        # bonds need to be faded in/out if they are not present in the last/next frame
        for i, row in enumerate(layout_data):
            rowdict = dict()
            for identifier, a, b in row.aucg_bonds:
                if identifier not in aucg_bonds.columns:
                    aucg_bonds[identifier] = np.NaN
                rowdict[identifier] = (a, b, identifier)
            # append returns a new object
            aucg_bonds = aucg_bonds.append(rowdict, ignore_index=True)

            rowdict = dict()
            for identifier, pos in row.gu_bonds:
                if identifier not in gu_bonds.columns:
                    gu_bonds[identifier] = np.NaN
                rowdict[identifier] = (pos, identifier)
            # append returns a new object
            gu_bonds = gu_bonds.append(rowdict, ignore_index=True)


        for k, c in enumerate(coordinates):
            # backbone = backbone_links.loc[[i]]
            # print (backbone)
            draw_objects = []

            # nucleotide characters
            # for index, value in backbone.items():
            for i, char in enumerate(sequence):

                (a, b), ch = annotation_chars[i][k]
                a *= scene_scalar  
                b *= scene_scalar  
                text = Text(ch, font_size=13)
                text.move_to([a, b, 0])
                draw_objects.append(text)

                if i+1 == len(sequence):
                    break

                # print (value[0])
                (a, b), (c, d) = backbone_links[i][k]
                a *= scene_scalar 
                b *= scene_scalar  
                c *= scene_scalar  
                d *= scene_scalar  

                line = Line((a, b, 0), (c, d, 0))
                line.stroke_width = 3
                line.set_color(GREEN)
                draw_objects.append(line)

            # annotation markers
            for ((x, y), ch), ((a, b), (c, d)) in zip(annotation_numbers[k], annotation_lines[k]):

                a *= scene_scalar  
                b *= scene_scalar  
                c *= scene_scalar  
                d *= scene_scalar  

                line = Line((a, b, 0), (c, d, 0))
                line.stroke_width = 1.5
                line.set_color(GREY)
                draw_objects.append(line)

                x *= scene_scalar  
                y *= scene_scalar  
                text = Text(str(ch), font_size=15)
                text.move_to([x, y, 0])
                draw_objects.append(text)

            # aucg bonds
            for index, value in aucg_bonds.items():

                current = value[k]
                if not isinstance(current, tuple):
                    # calculate fade in/out positions for bonds
                    pos1, pos2 = [int(i) for i in index.split("-")]

                    if pos2 == len(sequence):
                        pos2 = pos1

                    # a, b = (coordinates[k][pos1]+coordinates[k][pos2])/2
                    # c, d = (coordinates[k][pos1]+coordinates[k][pos2])/2
                    a, b = coordinates[k][pos1]*0.33+coordinates[k][pos2]*0.66
                    c, d = coordinates[k][pos1]*0.66+coordinates[k][pos2]*0.33

                    line_width = 0 # fade in/out: opacity to zero
                else:
                    # print (value[k])
                    (a, b), (c, d), identifier = value[k]
                    # print (a,b,c,d)
                    line_width = 2.5

                a *= scene_scalar  
                b *= scene_scalar  
                c *= scene_scalar  
                d *= scene_scalar 

                if a < c:
                    a, b, c, d = c, d, a, b

                line = Line((a, b, 0), (c, d, 0))
                line.stroke_width = line_width
                line.set_color(RED)
                draw_objects.append(line)

            # gu bonds
            for index, value in gu_bonds.items():

                current = value[k]
                if not isinstance(current, tuple):
                    pos1, pos2 = [int(i) for i in index.split("-")]
                    a, b = (coordinates[k][pos1] + coordinates[k][pos2])/2
                    line_width = 0

                else:
                    (a, b), identifier = value[k]
                    line_width = 1.5

                a *= scene_scalar  
                b *= scene_scalar  

                circle = Circle(radius=0.025, color=RED)
                circle.move_to((a, b, 0))
                circle.stroke_width = line_width
                draw_objects.append(circle)

            # render all objects
            draw_objects = VGroup(*draw_objects)


            if k == 0:  # initial fade in
                self.play(FadeIn(draw_objects))
                self.wait(fadeinout_seconds)
            elif k+1 != len(coordinates):
                self.play(Transform(last_objects, draw_objects, run_time=transition_seconds))
                self.wait(wait_seconds)
                self.remove(last_objects)
            else:  # last iteration
                self.play(Transform(last_objects, draw_objects, run_time=transition_seconds))
                self.wait(fadeinout_seconds)
                self.play(FadeOut(last_objects))

            last_objects = draw_objects

