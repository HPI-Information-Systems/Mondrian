import json
import ntpath
import os
import pickle

import PIL
import matplotlib
import networkx as nx
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
from matplotlib.patches import FancyArrowPatch

from scipy.stats import rankdata

from .mondrian import find_regions
from .region import Region
from .. import colors as colors
from ..visualization import table_as_image, draw_rectangles, draw_grid

DIR_STRINGS = {0: "N", 1: "H", 2: "V", 3: "O"}


class Spreadsheet:

    def __init__(self, file_path, delimiter=",", regions=False, partitioning=True, save_path=None, printing=False):

        self.file_path = file_path
        self.filename = ntpath.basename(file_path)

        self.layout = None
        self.clusters = None
        self.color_list = None
        self.similar_files = set([self.filename])

        try:
            region_path = save_path + self.filename + "_regions.json"
            empty_path = save_path + self.filename + "_empty_regions.json"
            self.regions = self.restore(region_path)
            self.empty_regions = self.restore(empty_path)

        except:
            if not regions:
                return
            color_img = table_as_image(file_path, delimiter=delimiter, color=True)
            self.regions = [Region(points_tuple=p, filename=self.filename, img=color_img)
                            for p in find_regions(self, partitioning=partitioning)]
            self.empty_regions = []

            if len(self.regions) > 1:
                self.empty_regions = [Region(points_tuple=p, filename=self.filename, img=color_img, type="empty")
                                      for p in find_regions(self, inverse_regions=self.regions, inverse=True)]
            if save_path:
                self.save(region_path, self.regions)
                self.save(empty_path, self.regions)

        if printing:
            im = Image.fromarray(color_img)
            im = im.resize(tuple(np.array(im.size) * 100))
            im.save("res/img/files/colored/" + self.filename + ".png")

            self.print_image()
            self.print_layout()

    def restore(self, path, pckl=False):
        color_img = table_as_image(self.file_path, color=True)
        if not pckl:
            with open(path, "r") as json_file:
                dict_json = json.load(json_file)
            return [Region.fromdict(r_dict, color_img) for k, r_dict in dict_json.items()]
        else:
            with open(path, "rb") as pckl_file:
                region_list, _, _ = pickle.load(pckl_file)
            return [Region(filename=self.filename, img=color_img, **r) for r in region_list]

    def save(self, json_path, region_objects):
        Path(os.path.split(json_path)[0]).mkdir(parents=True, exist_ok=True)
        out_json = {f"{idx}": r.asdict() for idx, r in enumerate(region_objects)}
        json.dump(out_json, open(json_path, "w"))

    def print_image(self, img, save_path=None):
        if self.clusters is not None:
            rectangles = self.clusters
        else:
            rectangles = self.regions

        FACTOR = 100
        im = Image.fromarray(img)
        im = im.resize(tuple(np.array(im.size) * FACTOR), resample=PIL.Image.NEAREST)
        draw = ImageDraw.Draw(im, "RGBA")
        _, self.color_list = draw_rectangles(draw, [{"top_lx": r.top_lx, "bot_rx": r.bot_rx} for r in rectangles],
                                             factor=FACTOR, alpha=200, width=3)

        try:
            _, color_empty = draw_rectangles(draw, [{"top_lx": r.top_lx, "bot_rx": r.bot_rx} for r in self.empty_regions],
                                             factor=FACTOR, alpha=200, width=3, color=colors.RGB_WHITE,
                                             out_color=colors.RGB_BLUE)
        except AttributeError:
            pass

        draw_grid(draw, img_size=np.shape(img),
                  factor=FACTOR, color=tuple(colors.RGB_BLACK), line_width=3,
                  text=True, text_color=tuple(colors.RGB_AQUA))

        if save_path is None:
            save_path = "res/img/files/"
        im.save(save_path + self.filename + ".png")

        fig = plt.figure()
        plt.title(self.filename)
        plt.imshow(im)
        ax = plt.gca()
        ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
        ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
        plt.show()

    def print_layout(self, save_path=None):

        if self.layout is None:
            print("Spreadsheet has no layout!")
            return

        n_empty = len([e for e in self.layout.nodes if self.layout.nodes[e]["region"].type == "empty"])
        if n_empty > 0:
            [self.color_list.append([*colors.RGB_WHITE, 200]) for n in range(n_empty)]
        color_list = np.asarray(self.color_list) / 255

        plt.figure(figsize=(10, 25))
        if len(self.layout.nodes) > 1:
            scale = 10
            pos = nx.spring_layout(self.layout)
            N = len(self.layout.nodes)

            nx.draw_networkx(self.layout, pos, with_labels=True, node_color = color_list,
                             node_shape="s", node_size=int(6000 / N),
                             font_color="black", font_size=int(scale / np.log10(N + 1)),
                             linewidths=2, width=1, edgecolors="black")

            direct = nx.get_edge_attributes(self.layout, 'direction')
            direct = {k: DIR_STRINGS[v] for k, v in direct.items()}
            weights = nx.get_edge_attributes(self.layout, 'weight')
            distance = nx.get_edge_attributes(self.layout, 'distance')
            labels = {k: f"{direct[k]}, {weights[k]}, {distance[k]}" for k in direct}
            nx.draw_networkx_edge_labels(self.layout, pos, edge_labels=labels, font_size=int(scale/np.log10(N+1)))

            plt.title(self.filename)
            plt.gca().invert_yaxis()
            if save_path is None:
                save_path = "res/img/graphs/"
            plt.savefig(save_path + self.filename + "_graph.png")
            plt.show()
