import itertools
import pickle

import networkx as nx
import numpy as np

import cv2 as cv
from PIL import Image
from PIL import ImageDraw

from joblib import Parallel, delayed
from sklearn.cluster import DBSCAN

from .region import Region, rectangles_from_lines, dominating, region_adjacency, region_distance
from ..clustering import cdist_generic, denoise_labels
from ..distances import parallel_distance
from ..visualization import table_as_image

'''
Class that is used to encapsulate the Mondrian algorithms.
Partitioning, clustering, etc. etc.
'''

RADIUS = 1
ALPHA = 1
BETA = 0.5
GAMMA = 10

def find_regions(spreadsheet, partitioning=True, inverse_regions=None, inverse=False):
    if inverse_regions is None:
        inverse_regions = []
    img = table_as_image(spreadsheet.file_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = 255 - img if not inverse else img
    img_height, img_width = np.shape(img)
    if inverse:
        im = Image.fromarray(img)
        draw = ImageDraw.Draw(im)
        for r in inverse_regions:
            draw.rectangle([tuple(r.top_lx), tuple(r.bot_rx)], fill=0)
        img = np.asarray(im)

    # prevent segfault of opencv FindCountours
    if (img == 0).all():
        return []
    contours, hierarchy = cv.findContours(img, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE, None, 1)
    # hierarchy is list of [Next, Previous, First_Child, Parent]
    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    list_sons = [None] * len(contours)
    for idx, _ in enumerate(contours):
        list_sons[idx] = [j for j, h in enumerate(hierarchy[0]) if h[3] == idx]

    regions = set()

    if not partitioning:
        for c in contours:
            x, y, width, height = cv.boundingRect(c)
            bbox = x, x + width - 1, y, y + height - 1
            regions.add(
                Region(points_tuple=bbox, filename=spreadsheet.filename, img=spreadsheet.color_img))
        return regions

    for idx, cnt in enumerate(contours):
        if hierarchy[0][idx][3] >= 0:  # exclude sons because they are holes
            continue
        detail_img = np.zeros(np.shape(img), np.uint8)  # this ensures no intersecting regions
        detail_img = Image.fromarray(detail_img)

        draw = ImageDraw.Draw(detail_img)
        drawable_cnt = list(np.append(cnt.flatten(), cnt.flatten()[0:2]))
        draw.polygon(drawable_cnt, fill=(255, 255, 255))

        if list_sons[idx]:
            for j in list_sons[idx]:
                drawable_cnt = list(np.append(contours[j].flatten(), contours[j].flatten()[0:2]))
                draw.polygon(drawable_cnt, fill=(0, 0, 0))  # empty sons
                draw.line(drawable_cnt, fill=(255, 255, 255))  # but contour is full
                cnt = np.vstack((cnt, contours[j]))

        detail_img = np.asarray(detail_img)
        detail_img = 255 - detail_img
        contour_lines = set()
        for p in set([tuple(x[0]) for x in cnt]):
            nbrs = neighbors(p, detail_img)
            x, y = p[0], p[1]
            if not nbrs[0]:  # left is empty
                contour_lines.add(("v", x))  # v_line_left
            if not nbrs[1]:  # bot is empty
                contour_lines.add(("h", y + 1))  # h_line_bot
            if not nbrs[2]:  # right is empty
                contour_lines.add(("v", x + 1))  # v_line_right
            if not nbrs[3]:  # top is empty
                contour_lines.add(("h", y))  # h_line_top

        h_lines = sorted([l for l in contour_lines if l[0] == "h"], key=lambda x: x[1])
        v_lines = sorted([l for l in contour_lines if l[0] == "v"], key=lambda x: x[1])
        h_pairs = list(zip(h_lines, h_lines[1:]))
        v_pairs = list(zip(v_lines, v_lines[1:]))

        cycles = list(itertools.product(h_pairs, v_pairs))
        cycles = [{*p[0], *p[1]} for p in cycles]

        cycles = set([frozenset(x) for x in cycles])
        partitions = [rectangles_from_lines(c) for c in cycles]

        # prune partitions
        if len(partitions) > 1:
            if list_sons[idx]:
                detail_img = 255 - img
            partitions = [p for p in partitions if (
                    detail_img[p[1]: p[3] + 1, p[0]: p[2] + 1]  # if roi is completely black
                    == [0, 0, 0]).all(axis=2).all()]

            before_dom = len(partitions)
            to_delete = Parallel(n_jobs=-1)(delayed(dominating)(a, b) for a, b in itertools.combinations(partitions, 2))
            to_delete = [d for d in to_delete if d is not None]
            [partitions.remove(d) for d in set(to_delete)]

        regions |= set(partitions)

    return regions

def neighbors(pixel, img):
    '''
    Function used to check for a given pixel the content of neighboring pixels.
    '''

    height, width, _ = np.shape(img)
    x, y = pixel
    if x - 1 >= 0:
        n_left = int(all(img[y, x - 1] == [0, 0, 0]))
    else:
        n_left = 0
    try:
        n_right = int(all(img[y, x + 1] == [0, 0, 0]))
    except IndexError:
        n_right = 0
    if y - 1 >= 0:
        n_top = int(all(img[y - 1, x] == [0, 0, 0]))
    else:
        n_top = 0
    try:
        n_bot = int(all(img[y + 1, x] == [0, 0, 0]))
    except IndexError:
        n_bot = 0

    return [n_left, n_bot, n_right, n_top]



def merge_nodes(spreadsheet, G, list_nodes):
    while len(list_nodes) > 0: 
        e = list_nodes[0]
        merged = False
        for v in G[e]:
            if G.nodes[v]["region"].type == G.nodes[e]["region"].type:
                # if they have no shared neighbors OR any of the shared neighbors has same direction
                shared_neighbors = set(G[v]).intersection(set(G[e]))
                if shared_neighbors == {} or all(
                        [G[v][n]["direction"] == G[e][n]["direction"] for n in shared_neighbors]):
                    new_top_lx = [min(G.nodes[v]["region"].top_lx[0], G.nodes[e]["region"].top_lx[0]),
                                  min(G.nodes[v]["region"].top_lx[1], G.nodes[e]["region"].top_lx[1])]
                    new_bot_rx = [max(G.nodes[v]["region"].bot_rx[0], G.nodes[e]["region"].bot_rx[0]),
                                  max(G.nodes[v]["region"].bot_rx[1], G.nodes[e]["region"].bot_rx[1])]
                    new_node = Region(points_tuple=(*new_top_lx, *new_bot_rx),
                                      filename=G.nodes[e]["region"].filename,
                                      img=spreadsheet.color_img,
                                      type=G.nodes[e]["region"].type)

                    G.add_node(str(new_node), region=new_node)
                    for n in set(G[v]) | set(G[e]):
                        d, w = region_adjacency(G.nodes[n]["region"], new_node)
                        if w != 0:
                            G.add_edge(n, str(new_node), direction=d, weight=w)

                    list_nodes.append(str(new_node))
                    try:
                        list_nodes.remove(v)
                    except ValueError:
                        pass  # this means it was previously inspected
                    list_nodes.remove(e)
                    G.remove_node(v)
                    G.remove_node(e)
                    merged = True
                    break
                else:
                    continue
        if not merged:
            list_nodes = list_nodes[1:]

    return G

def calculate_layout(region_objects, save_path = None, overwrite= ""):
    G = nx.Graph()
    G.add_nodes_from([(str(r), {"region": r}) for r in region_objects])

    try:
        edges = pickle.load(open(save_path+overwrite, "rb"))
    except:
        rpairs = list(itertools.combinations(nx.get_node_attributes(G,"region").values(),2))
        edges = list(itertools.starmap(region_distance, rpairs))
        edges = [(r1,r2, edges[idx]) for idx,(r1,r2) in enumerate(itertools.combinations(G.nodes,2))]
        if save_path:
            pickle.dump(edges, open(save_path, "wb"))
    G.add_edges_from([e for e in edges if e[2]["weight"] != 0])
    return G

class Mondrian:

    def __init__(self, alpha=ALPHA, beta=BETA, gamma=GAMMA, radius=RADIUS):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.radius = radius

    def cluster_regions(self, spreadsheet, min_samples=1, n_jobs=-1):
        p_regions = np.asarray([r.asarray() for r in spreadsheet.regions])
        img = table_as_image(spreadsheet.file_path)
        height, width, _ = np.shape(img)

        distances = cdist_generic(dist_fun=parallel_distance, dataset1=p_regions, n_jobs=n_jobs,
                                  file_width=width, file_height=height, alpha=self.alpha,
                                  beta=self.beta,
                                  gamma=self.gamma)

        db = DBSCAN(eps=self.radius, min_samples=min_samples, metric="precomputed", n_jobs=n_jobs)
        db.fit(distances)

        labels = denoise_labels(db.labels_)

        # find cluster edges
        tmp = {"top_lx": [float("inf"), float("inf")], "bot_rx": [-1, -1]}
        n_clusters = len(set(labels))
        cluster_edges = [tmp for _ in range(n_clusters)]

        for idx, r in enumerate(spreadsheet.regions):
            cluster = labels[idx]
            current = cluster_edges[cluster]

            cluster_edges[cluster] = {"top_lx": np.minimum(current["top_lx"], r.top_lx).astype(int),
                                      "bot_rx": np.maximum(current["bot_rx"], r.bot_rx).astype(int)}

        return [(*e["top_lx"], *e["bot_rx"]) for e in cluster_edges]
