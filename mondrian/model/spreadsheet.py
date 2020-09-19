import functools
import itertools
import json
import ntpath
import time

import matplotlib
import networkx as nx
from networkx.drawing.layout import rescale_layout, planar_layout
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import write_dot
from scipy.interpolate import interp1d
from scipy.stats import rankdata

from mondrian.model.mondrian import *
from mondrian.model.region import *
from mondrian.visualization import *
from mondrian.clustering import *
import mondrian.colors as colors


class Spreadsheet:

    def __init__(self, file_path, delimiter=",", regions = False, partitioning=True, save_path=None, printing=False):

        self.file_path = file_path
        self.filename = ntpath.basename(file_path)

        self.layout = None
        self.clusters = None
        self.color_list = None
        self.similar_files = set([self.filename])

        # self.img = table_as_image(file_path, delimiter=delimiter)
        # self.color_img = table_as_image(file_path, delimiter=delimiter, color=True)
        # self.height, self.width, _ = np.shape(self.img)

        try:
            region_path = save_path + self.filename + "_regions.json"
            empty_path = save_path + self.filename + "_empty_regions.json"  # TODO Remove empty regions altogether?
            self.regions = self.restore(region_path)
            self.empty_regions = self.restore(empty_path)

        except:
            if not regions:
                return
            color_img = table_as_image(file_path, delimiter=delimiter, color=True)
            self.regions = [Region(points_tuple=p, filename=self.filename, img=color_img)
                            for p in find_regions(self,partitioning=partitioning)]
            self.empty_regions = []

            if len(self.regions) > 1:
                self.empty_regions = [Region(points_tuple=p, filename=self.filename, img=color_img, type="empty")
                                      for p in find_regions(self, inverse_regions=self.regions, inverse=True)]
            if save_path:
                self.save(region_path, self.regions)
                self.save(empty_path, self.regions)

        # del self.img, self.color_img

        if printing:
            im = Image.fromarray(color_img)
            im = im.resize(tuple(np.array(im.size) * 100))
            im.save("res/img/files/colored/" + self.filename + ".png")

            self.print_image()
            self.print_layout()

    def restore(self, path, pckl=False):
        color_img = table_as_image(self.file_path, color=True)
        if not pckl:
            with open(path,"r") as json_file:
                dict_json = json.load(json_file)
            return [Region.fromdict(r_dict, color_img) for k, r_dict in dict_json.items()]
        else:
            with open(path, "rb") as pckl_file:
                region_list, _, _ = pickle.load(pckl_file)
            return [Region(filename = self.filename,img= color_img, **r) for r in region_list]


    def save(self, json_path, region_objects):
        Path(os.path.split(json_path)[0]).mkdir(parents=True, exist_ok=True)
        out_json = {f"{idx}": r.asdict() for idx, r in enumerate(region_objects)}
        json.dump(out_json, open(json_path, "w"))

    def print_image(self):
        if self.clusters is not None:
            rectangles = self.clusters
        else:
            rectangles = self.regions

        FACTOR = 100
        im = Image.fromarray(self.img)
        im = im.resize(tuple(np.array(im.size) * FACTOR))
        draw = ImageDraw.Draw(im, "RGBA")
        _, self.color_list = draw_rectangles(draw, [{"top_lx": r.top_lx, "bot_rx": r.bot_rx} for r in rectangles],
                                             factor=FACTOR, alpha=200, width=3)

        _, color_empty = draw_rectangles(draw, [{"top_lx": r.top_lx, "bot_rx": r.bot_rx} for r in self.empty_regions],
                                         factor=FACTOR, alpha=200, width=3, color=colors.RGB_WHITE,
                                         out_color=colors.RGB_BLUE)

        draw_grid(draw, img_size=np.shape(self.img),
                  factor=FACTOR, color=tuple(colors.RGB_BLACK), line_width=3,
                  text=True, text_color=tuple(colors.RGB_AQUA))

        im.save("res/img/files/" + self.filename + ".png")

        fig = plt.figure()
        plt.imshow(im)
        ax = plt.gca()
        ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
        ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())

    def print_layout(self):

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
            x = rankdata([r.top_lx[0] for k, r in nx.get_node_attributes(self.layout, "region").items()],
                         method='min') - 1
            y = rankdata([r.top_lx[1] for k, r in nx.get_node_attributes(self.layout, "region").items()],
                         method='min') - 1

            pos_x = [scale * (2 * x[i] - 1) for i, _ in enumerate(x)]
            pos_y = [scale * (2 * y[i] - 1) for i, _ in enumerate(y)]
            pos = {k: [pos_x[i], pos_y[i]] for i, (k, r) in
                   enumerate(nx.get_node_attributes(self.layout, "region").items())}

            N = len(self.layout.nodes)
            nx.draw_networkx(self.layout, pos,
                             node_shape="s", node_size=int(60000 / N),
                             font_size=int(25 / np.log10(N + 1)),
                             font_color="black", node_color=color_list,
                             linewidths=2, edgecolors="black")
            direct = nx.get_edge_attributes(self.layout, 'direction')
            weights = nx.get_edge_attributes(self.layout, 'weight')
            labels = {k: direct[k] + str(weights[k]) for k in direct}
            nx.draw_networkx_edge_labels(self.layout, pos, edge_labels=labels, font_size=35)

            plt.title(self.filename)
            xmargin = scale * 5
            ymargin = scale
            plt.axis((min(pos_x) - xmargin, max(pos_x) + xmargin, min(pos_y) - ymargin, max(pos_y) + ymargin))
            plt.gca().invert_yaxis()
            plt.margins(0.3)
            plt.savefig("res/img/graphs/" + self.filename + ".png")
            plt.show()
