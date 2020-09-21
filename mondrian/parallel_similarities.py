from __future__ import print_function
import builtins as __builtin__
from scipy.sparse import coo_matrix
from scipy.special._ufuncs import binom

from multiprocessing.pool import Pool
import itertools

import numpy as np
import networkx as nx
import pandas as pd
import time
from scipy.optimize import linear_sum_assignment

from .model.region import parallel_region_sim, DIRECTION_NONE


def print(*args, **kwargs):
    return __builtin__.print(f"\033[94m{time.process_time()}:\033[0m", *args, **kwargs)

def parallel_neigh_props(m, n, i, j, edge_prop):
    indices_x = [int(i + (x * m) - sum(range(x + 1))) for x in range(i + 1)]
    indices_x += list(range(max(indices_x) + 1, max(indices_x) + m - len(indices_x) + 1))
    indices_x = np.asarray(indices_x)

    indices_y = [int(j + (y * n) - sum(range(y + 1))) for y in range(j + 1)]
    indices_y += list(range(max(indices_y) + 1, max(indices_y) + n - len(indices_y) + 1))
    indices_y = np.asarray(indices_y)

    return edge_prop[indices_y,indices_x[:, None]]

def vectorized_edge_sim(direction_a, direction_b, weight_a, weight_b, distance_a, distance_b):
    z = np.array([DIRECTION_NONE] *len(direction_a))
    nonzero = (direction_a == direction_b) & (direction_a != z) & (direction_b != z)  # only consider where directions are equal and not none

    w_a = np.array([weight_a, distance_a])
    w_b = np.array([weight_b, distance_b])

    m = np.max([w_a, w_b], axis=0)
    m[m == 0] = 1
    sim = 1 - np.linalg.norm(w_a / m - w_b / m, axis=0) / np.sqrt(2)  # sqrt(2) is max achievable
    return nonzero * sim

def parallel_unisim(s0, edges_prop, n_jobs=1, verbose = False):
    node_sim = parallel_similarity_flooding(s0, edges_prop, n_jobs=n_jobs, verbose = verbose)
    mwm = linear_sum_assignment(1 - node_sim)  # function minimizes weights
    node_match = list(zip(mwm[0], mwm[1]))
    node_diff = np.abs(np.diff(s0.shape)[0])
    return np.average([node_sim[i, j] for i, j in node_match] + [0] * node_diff)

def parallel_update_sim(sim_ij, s0_ij, s0, sim, local_edge_prop):
    return sim_ij * (s0_ij + sum(np.max((s0 + sim) * local_edge_prop.toarray(), axis=1)))

def parallel_similarity_flooding(s0, edge_prop, SIM_EPS=0.001, N_ITER=1000, n_jobs=1, verbose = False):
    m, n = np.shape(s0)
    sim = s0

    terminate = False
    n_iter = 0

    args = [(x, y, *z, t) for x, y, z, t in zip([m] * (m * n), [n] * (m * n), itertools.product(range(m), range(n)), [edge_prop] * (m * n))]

    if verbose: print("Calculating local edges matrix")
    if n_jobs > 1:
        with Pool(n_jobs) as pool:
            local_edge_prop = list(pool.starmap(parallel_neigh_props, args, chunksize = int(len(args)/n_jobs)))
    else:
        local_edge_prop = list(itertools.starmap(parallel_neigh_props, args))

    while not terminate:
        args = [(sim[i, j], s0[i, j], s0, sim, local_edge_prop[i + j * m]) for i in range(m) for j in range(n)]
        if n_jobs > 1:
            with Pool(n_jobs) as pool:
                new_sim = list(pool.starmap(parallel_update_sim, args, chunksize=int(len(args) / n_jobs)))
        else:
            new_sim = list(itertools.starmap(parallel_update_sim, args))

        new_sim = np.array(new_sim).reshape(s0.shape)
        norm_sim = new_sim
        norm_sim /= np.max(new_sim, axis=0)  # normalize by columns
        norm_sim[np.isnan(norm_sim)] = 0
        delta_sim = np.linalg.norm(norm_sim - sim)
        if verbose: print(f"Iteration {n_iter} delta was {delta_sim}")
        if delta_sim < SIM_EPS:
            if verbose: print(f"\tTerminating for delta d={delta_sim}, i was {n_iter}")
            terminate = True
        if n_iter >= N_ITER:
            if verbose: print(f"\tTerminating for i={n_iter}, d was {delta_sim}")
            terminate = True
        sim = norm_sim
        n_iter += 1
    return sim * s0


def parallel_layout_similarity(layout_a, layout_b, n_jobs=None, verbose = False, allthresholds = False):
    m, n = len(layout_a.nodes), len(layout_b.nodes)
    if not allthresholds:
        upper_boundary = min(m, n) / max(m, n)
        if upper_boundary < 0.7:
            return 0

    if n_jobs is None:
        n_jobs = 1

    regions_a = [x for k, x in nx.get_node_attributes(layout_a, "region").items()]
    regions_b = [x for k, x in nx.get_node_attributes(layout_b, "region").items()]
    index_pairs = itertools.product(regions_a, regions_b)
    index_pairs = [(a.color_hist, b.color_hist) for a, b in index_pairs]

    if verbose: print("Calculating s0")
    if n_jobs > 1:
        chunk_size = int(len(index_pairs) / n_jobs)
        with Pool(n_jobs) as pool:
            s0 = list(pool.starmap(parallel_region_sim, index_pairs, chunksize=chunk_size))
    else:
        s0 = list(itertools.starmap(parallel_region_sim, index_pairs))
    if verbose: print("Finished calculating s0")

    s0 = np.array(s0).reshape((m, n))

    if verbose: print("Generating edge similarity pairs...")
    regions_a_df = pd.DataFrame({"region": [str(r) for r in regions_a]})
    regions_a_df["idx"] = regions_a_df.index
    regions_b_df = pd.DataFrame({"region": [str(r) for r in regions_b]})
    regions_b_df["idx"] = regions_b_df.index

    default_edge = {"direction": DIRECTION_NONE, "weight": 0, "distance": 0}
    edges_a_df = pd.DataFrame([{"r1": str(r1), "r2": str(r2), **layout_a.get_edge_data(str(r1), str(r2), default_edge)}
                               for r1, r2 in itertools.combinations_with_replacement(regions_a, 2)])
    edges_b_df = pd.DataFrame([{"r1": str(r1), "r2": str(r2), **layout_b.get_edge_data(str(r1), str(r2), default_edge)}
                               for r1, r2 in itertools.combinations_with_replacement(regions_b, 2)])
    edges_a_df = edges_a_df.merge(regions_a_df, left_on="r1", right_on="region", how="left")
    edges_a_df.drop(columns=["r1", "region"], inplace=True)
    edges_a_df.rename(columns={"idx": "r1"}, inplace=True)
    edges_a_df = edges_a_df.merge(regions_a_df, left_on="r2", right_on="region", how="left")
    edges_a_df.drop(columns=["r2", "region"], inplace=True)
    edges_a_df.rename(columns={"idx": "r2"}, inplace=True)

    edges_b_df = edges_b_df.merge(regions_b_df, left_on="r1", right_on="region", how="left")
    edges_b_df.drop(columns=["r1", "region"], inplace=True)
    edges_b_df.rename(columns={"idx": "r1"}, inplace=True)
    edges_b_df = edges_b_df.merge(regions_b_df, left_on="r2", right_on="region", how="left")
    edges_b_df.drop(columns=["r2", "region"], inplace=True)
    edges_b_df.rename(columns={"idx": "r2"}, inplace=True)

    edges_a_df["joinkey"] = 0
    edges_b_df["joinkey"] = 0
    if verbose: print("Calculating edge propagation matrix...")
    t1 = edges_a_df.query(f"direction != {DIRECTION_NONE}")
    t2 = edges_b_df.query(f"direction != {DIRECTION_NONE}")
    edge_sims_df = t1.merge(t2, on="joinkey", how="inner", suffixes=["_a", "_b"])

    edge_sims_df["similarity"] = vectorized_edge_sim(edge_sims_df.direction_a.values, edge_sims_df.direction_b.values,
                                                     edge_sims_df.weight_a.values, edge_sims_df.weight_b.values,
                                                     edge_sims_df.distance_a.values, edge_sims_df.distance_b.values)
    cols_df = pd.DataFrame([{"r1_a":r1, "r2_a":r2, "col_idx":idx} for idx, (r1,r2) in enumerate(itertools.combinations_with_replacement(range(m), 2))])
    rows_df = pd.DataFrame([{"r1_b":r1, "r2_b":r2, "row_idx":idx} for idx, (r1,r2) in enumerate(itertools.combinations_with_replacement(range(n), 2))])
    edge_sims_df = edge_sims_df.merge(cols_df)
    edge_sims_df = edge_sims_df.merge(rows_df)

    m2, n2 = int(binom(m+1,2)), int(binom(n+1,2))
    edge_prop = coo_matrix((edge_sims_df.similarity.values, (edge_sims_df.row_idx.values, edge_sims_df.col_idx.values)), shape=(n2, m2))
    edge_prop = edge_prop.tocsr()

    a_sim_b = parallel_unisim(s0, edge_prop, verbose = verbose, n_jobs=n_jobs)
    if 1 in s0.shape:
        b_sim_a = a_sim_b
    else:
        b_sim_a = parallel_unisim(s0.transpose(), edge_prop.transpose(), verbose = verbose, n_jobs=n_jobs)

    avg_sim = np.average([a_sim_b, b_sim_a])
    return avg_sim
