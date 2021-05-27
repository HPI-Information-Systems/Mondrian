from __future__ import print_function
import builtins as __builtin__
import time

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import itertools

import numpy as np
import networkx as nx
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment

from .region import parallel_region_sim, DIRECTION_NONE


def print(*args, **kwargs):
    return __builtin__.print(f"\033[94m{time.process_time()}:\033[0m", *args, **kwargs)


def edge_similarity(edge_a, edge_b):
    w_a = np.asarray([edge_a["weight"], edge_a["distance"]])
    w_b = np.asarray([edge_b["weight"], edge_b["distance"]])
    dist = int(edge_a["direction"] == edge_b["direction"])
    dist *= np.linalg.norm(w_a - w_b) / np.max(np.abs(np.append(w_a, w_b)))

    return 1 - dist


def neigh_props(m, n, i, j):
    indices_x = [int(i + (x * m) - sum(range(x + 1))) for x in range(i + 1)]
    indices_x += list(range(max(indices_x) + 1, max(indices_x) + m - len(indices_x) + 1))
    indices_x = np.asarray(indices_x)

    indices_y = [int(j + (y * n) - sum(range(y + 1))) for y in range(j + 1)]
    indices_y += list(range(max(indices_y) + 1, max(indices_y) + n - len(indices_y) + 1))
    indices_y = np.asarray(indices_y)

    return (indices_x[:, None], indices_y)


def parallel_matrix_indices(pair_obj):
    a, b, layout_a, layout_b = pair_obj
    default_edge = {"direction": DIRECTION_NONE, "weight": 0, "distance": 0}
    return (layout_a.get_edge_data(*pair[0], default_edge), layout_b.get_edge_data(*pair[1], default_edge))


def edge_dist(p):
    edge_a, edge_b = p[0], p[1]
    if edge_a["direction"] == DIRECTION_NONE or edge_b["direction"] == DIRECTION_NONE:
        return 0
    w_a = np.asarray([edge_a["weight"], edge_a["distance"]])
    w_b = np.asarray([edge_b["weight"], edge_b["distance"]])
    dist = int(edge_a["direction"] == edge_b["direction"])
    dist *= np.linalg.norm(w_a - w_b) / np.max(np.abs(np.append(w_a, w_b)))

    return 1 - dist


def propagation_func(p):
    edges_a, edges_b, u, n_u, v, n_v = p[2], p[3], p[0][0], p[0][1], p[1][0], p[1][1]
    try:
        edge_a = edges_a[u, n_u]
        edge_b = edges_b[v, n_v]
    except:
        return 0

    w_a = np.asarray([edge_a["weight"], edge_a["distance"]])
    w_b = np.asarray([edge_b["weight"], edge_b["distance"]])

    m = np.max([w_a, w_b], axis=0)
    m[m == 0] = 1
    sim = 1 - np.linalg.norm(w_a / m - w_b / m, axis=0) / np.sqrt(2)  # sqrt(2) is max achievable

    return sim


def unisim(s0, edge_propagation, n_jobs=1):
    node_sim = similarity_flooding(s0, edge_propagation, n_jobs=n_jobs)
    mwm = linear_sum_assignment(1 - node_sim)  # function minimizes weights
    node_match = list(zip(mwm[0], mwm[1]))

    node_diff = np.abs(np.diff(s0.shape)[0])
    return np.average([node_sim[i, j] for i, j in node_match] + [0] * node_diff)


def similarity_flooding(s0, edge_propagation, SIM_EPS=0.1, N_ITER=10, n_jobs=1):
    m, n = np.shape(s0)
    sim = s0

    terminate = False
    n_iter = 0

    list_edge_neigh = [edge_propagation[neigh_props(m, n, i, j)] for i in range(m) for j in range(n)]
    if n_jobs > 1:
        print("Begin Flooding")
    while not terminate:

        n_works = n_jobs
        list_pairs = [(sim[i, j], s0[i, j], s0, sim, list_edge_neigh[i + j * m]) for i in range(m) for j in range(n)]

        list_works = np.array_split(list_pairs, n_works)
        new_sim = Parallel(n_jobs=n_jobs)(delayed(
            lambda pair_list: [p[0] * (p[1] + sum(np.max((p[2] + p[3]) * p[4], axis=1))) for p
                               in pair_list])(w) for w in list_works)
        new_sim = [x for lst in new_sim for x in lst]
        new_sim = np.array(new_sim).reshape(s0.shape)
        norm_sim = new_sim
        norm_sim /= np.max(new_sim, axis=0)  # normalize by columns
        norm_sim[np.isnan(norm_sim)] = 0
        delta_sim = np.linalg.norm(norm_sim - sim)
        if delta_sim < SIM_EPS or n_iter >= N_ITER:
            terminate = True
        sim = norm_sim
        n_iter += 1
    return sim * s0

def layout_similarity(layout_a, layout_b):
    m, n = len(layout_a.nodes), len(layout_b.nodes)
    upper_boundary = min(m, n) / max(m, n)
    if upper_boundary < 0.7:
        return 0

    regions_a = [x for k, x in nx.get_node_attributes(layout_a, "region").items()]
    regions_b = [x for k, x in nx.get_node_attributes(layout_b, "region").items()]
    index_pairs = itertools.product(regions_a, regions_b)
    index_pairs = [(a.color_hist, b.color_hist) for a, b in index_pairs]
    s0 = list(itertools.starmap(parallel_region_sim, index_pairs))
    s0 = np.array(s0).reshape((m, n))

    default_edge = {"direction": DIRECTION_NONE, "weight": 0, "distance": 0}
    pairs_a = [(str(r1), str(r2), default_edge) for r1, r2 in itertools.combinations_with_replacement(regions_a, 2)]
    pairs_b = [(str(r1), str(r2), default_edge) for r1, r2 in itertools.combinations_with_replacement(regions_b, 2)]
    matrix_node_indices = [(layout_a.get_edge_data(*a), layout_b.get_edge_data(*b)) for a, b in itertools.product(pairs_a, pairs_b)]
    edge_propagation = list(map(edge_dist, matrix_node_indices))

    del regions_a, regions_b
    edge_propagation = np.array(edge_propagation).reshape((len(pairs_a), len(pairs_b)))

    a_sim_b = unisim(s0, edge_propagation)
    if 1 in s0.shape:
        b_sim_a = a_sim_b
    else:
        b_sim_a = unisim(s0.transpose(), edge_propagation.transpose())

    avg_sim = np.average([a_sim_b, b_sim_a])
    return avg_sim

class FileTemplate:

    def __init__(self, spreadsheet):
        self.hashname = spreadsheet.filename + "_template"
        self.layout = spreadsheet.layout
        self.file_list = [spreadsheet.filename]

    def recompute_layout(self):
        self.layout = []
