import itertools, multiprocessing
import time

import numpy as np
from joblib import Parallel, delayed


def edge_similarity(edge_a, edge_b):
    w_a = np.asarray([edge_a["weight"], edge_a["distance"]])
    w_b = np.asarray([edge_b["weight"], edge_b["distance"]])
    dist = int(edge_a["direction"] == edge_b["direction"])
    dist *= np.linalg.norm(w_a - w_b) / np.max(np.abs(np.append(w_a, w_b)))

    return 1 - dist


def propagation_func2(edges_a, edges_b, u, n_u, v, n_v):
        try:
            edge_a = edges_a[u,n_u]
            edge_b = edges_b[v,n_v]
        except:
            return 0

        w_a = np.asarray([edge_a["weight"], edge_a["distance"]])
        w_b = np.asarray([edge_b["weight"], edge_b["distance"]])
        dist = int(edge_a["direction"] == edge_b["direction"])
        dist *= np.linalg.norm(w_a - w_b) / np.max(np.abs(np.append(w_a, w_b)))

        return 1 - dist


def matrix_propagation(i,j, s0, sim,edge_neigh):
    phi = s0[i, j]
    mat_phi = (s0 + sim) * edge_neigh
    phi += sum(np.max(mat_phi, axis=1))
    return sim[i, j] * phi



# NOT PARALELL ca 10 min
# layout_sims = {(f1,f2):layout_similarity(file_db[f1].layout,file_db[f2].layout) for f1,f2 in tqdm(layout_pairs)}

# PARALLEL WITHOUT LIST WORKS
# sims = Parallel(n_jobs=1)(delayed(
#     lambda f1, f2: {(f1, f2): layout_similarity(file_db[f1].layout, file_db[f2].layout)})
#                                 (f1, f2) for f1, f2 in layout_pairs)
# layout_sims = {k: v for d in sims for k, v in d.items()}

# PARALLEL WITH LIST WORKS FOR EASY PAIRS
# SEQUENTIAL APPROACH
# layout_pairs = [set([f1,f2]) for f1,f2 in layout_pairs]
# i = len(layout_pairs)-1
# while i>0:
#     f1,f2 = layout_pairs[i]
#     layout_pairs.remove(set([f1,f2]))
#     sim = layout_similarity(file_db[f1].layout, file_db[f2].layout)
#     if sim > TEMPLATE_THRESHOLD:
#         file_db[f1].similar_files.add(f2)
#         file_db[f2].similar_files.add(f1)
#         for f in file_db[f1].similar_files:
#             file_db[f].similar_files.add(f2)
#         for f in file_db[f2].similar_files:
#             file_db[f].similar_files.add(f1)
#     elif sim <= 0.8:
#         [layout_pairs.remove(set([f,f2])) for f in file_db[f1].similar_files if set([f,f2]) in layout_pairs]
#         [layout_pairs.remove(set([f,f1])) for f in file_db[f2].similar_files if set([f,f1]) in layout_pairs]
#
#     # layout_pairs = [p for p in layout_pairs if f2 not in p]
#     i = len(layout_pairs)-1
#     print(f"There are {i} pairs left to examine")


#----
# for file, s in file_db.items():
#     template_candidates = set()
#     for k, r in nx.get_node_attributes(s.layout, "region").items():
#         [template_candidates.add(f) for f in similar_regions[(r.filename,str(r))]]
#
#     l = s.layout
#     # print("File candidates:")
#     for candidate_file in template_candidates:
#         t = file_db[candidate_file]
#         # print("\tMatching with template", t.hashname.split("_")[0])
#         sim = layout_similarity(l, t.layout)
#         # print(f"\tSimilarity: {sim}")
#         if sim > TEMPLATE_THRESHOLD:  # is a distance
#             s.similar_files.add(candidate_file)
#             t.similar_files.add(s.filename)
#             for f in s.similar_files:
#                 file_db[f].similar_files.add(candidate_file)
#             for f in t.similar_files:
#                 file_db[f].similar_files.add(s.filename)




def similarity_flooding(graph_a, graph_b, s0, edge_propagation=None, SIM_EPS=0.1, N_ITER=10):
    n, m = len(graph_a.nodes), len(graph_b.nodes)

    a_idx = {x: idx for idx, x in enumerate(graph_a.nodes)}
    b_idx = {x: idx for idx, x in enumerate(graph_b.nodes)}

    nodes_cross = list(itertools.product(graph_a.nodes, graph_b.nodes))

    n_cores = multiprocessing.cpu_count()
    sim = s0

    edge_propagation = {}
    if len(nodes_cross) > 0:
        print("Going for parallel")
        for u, v in nodes_cross:
            neighbors_u = list(graph_a.neighbors(u))
            neighbors_v = list(graph_b.neighbors(v))
            nbr_diff = int(max(30, np.abs(len(neighbors_u) - len(
                neighbors_v))))  # normalize edge similarity by number of neighbors CAST TO INT TO AVOID 0
            edge_prop = Parallel(n_jobs=n_cores)(
                delayed(propagation_func2)(graph_a.edges[u, n_u], graph_b.edges[v, n_v], u, v, n_u, n_v, nbr_diff)
                for n_u, n_v in itertools.product(neighbors_u, neighbors_v))

            [edge_propagation.update(d) for d in edge_prop if d is not None]

    # expensive (?!)
    else:
        for u, v in nodes_cross:
            u_nbrs = len([x for x in graph_a.neighbors(u)])
            v_nbrs = len([x for x in graph_b.neighbors(v)])
            nbr_diff = int(
                np.abs(u_nbrs - v_nbrs))  # normalize edge similarity by number of neighbors CAST TO INT TO AVOID 0
            for n_u in graph_a.neighbors(u):
                for n_v in graph_b.neighbors(v):
                    if nbr_diff > 30:  # even with edge sim of 1, result would be <10^(-10)
                        edge_propagation.update({(u, v, n_u, n_v): 0})
                    else:
                        edge_propagation.update(
                            {(u, v, n_u, n_v): edge_similarity(graph_a.edges[u, n_u],
                                                               graph_b.edges[v, n_v]) / 2 ** nbr_diff})

    terminate = False
    n_iter = 0
    while not terminate:
        new_sim = np.zeros((n, m))
        for i, u in enumerate(graph_a.nodes):
            for j, v in enumerate(graph_b.nodes):
                # basic phi
                # phi = sum([sim[a_idx[n_u], b_idx[n_v]] * edge_propagation[(u, v, n_u, n_v)]
                #            for n_u in graph_a.neighbors(u) for n_v in graph_b.neighbors(v)])
                # phi C : s0 + si + phi(s0+s1)
                phi = s0[i, j]
                for n_u in graph_a.neighbors(u):
                    neighbor_sim = [0]
                    for n_v in graph_b.neighbors(v):
                        try:
                            neighbor_sim.append(
                                (s0[a_idx[n_u], b_idx[n_v]] + sim[a_idx[n_u], b_idx[n_v]]) * edge_propagation[
                                    (u, v, n_u, n_v)])
                        except:
                            print("here")
                    phi += max(neighbor_sim)  # select only the highes similar node
                # phi /= len(list(graph_a.neighbors(u)))
                new_sim[i, j] = sim[i, j] * phi

        norm_sim = new_sim / np.max(new_sim, axis=0)  # normalize by columns

        delta_sim = np.linalg.norm(norm_sim - sim)
        if delta_sim < SIM_EPS or n_iter >= N_ITER:
            # print(f"\tTerminating with d={delta_sim} and i={n_iter}")
            terminate = True
        sim = norm_sim
        n_iter += 1
    return sim * s0


# def node_similarity(node_a, node_b):
#     return region_similarity(node_a["region"], node_b["region"]) > REGION_THRESHOLD

def propagation_func(edges_a, edges_b, neighbors_u, neighbors_v, u, v):
    u_nbrs = len(neighbors_u)
    v_nbrs = len(neighbors_v)
    return_dict = {}
    nbr_diff = int(
        max(30, np.abs(u_nbrs - v_nbrs)))  # normalize edge similarity by number of neighbors CAST TO INT TO AVOID 0
    for n_u, n_v in itertools.product(neighbors_u, neighbors_v):
        if nbr_diff > 30:  # even with edge sim of 1, result would be <10^(-10)
            return_dict.update({(u, v, n_u, n_v): 0})
        else:
            return_dict.update(
                {(u, v, n_u, n_v): edge_similarity(edges_a[u, n_u], edges_b[v, n_v]) / 2 ** nbr_diff})
    return return_dict


def propagation_func2(edge_u, edge_v, u, v, n_u, n_v, nbr_diff):
    if nbr_diff > 30:  # even with edge sim of 1, result would be <10^(-10)
        return {(u, v, n_u, n_v): 0}
    else:
        return {(u, v, n_u, n_v): edge_similarity(edge_u, edge_v) / 2 ** nbr_diff}


##S0 - edgeprop
    # with Parallel(n_jobs=n_jobs, verbose=0, timeout=timeout) as parallel:
        # s0 = parallel(
        #     delayed(
        #         # lambda x: [region_similarity(regions_a[p[0]], regions_b[p[1]]) for p in x])(list_pairs) for list_pairs in list_works
        #         # lambda x: [region_similarity(p[0], p[1]) for p in x])(list_pairs) for list_pairs in list_works
        #         lambda x: [parallel_region_sim(p) for p in x])(list_pairs) for list_pairs in list_works
        # )
        # s0 = [x for lst in s0 for x in lst]

#edgeprop
        # t_a = list(itertools.combinations_with_replacement(range(m), 2))
        # t_b = list(itertools.combinations_with_replacement(range(m, m + n), 2))
        # matrix_node_indices = list(itertools.product(t_a, t_b))
        # all_nodes = [ str(a) for a in regions_a] + [str(b) for b in regions_b]

        # matrix_node_indices = [(a,b, layout_a.edges, layout_b.edges) for a,b in itertools.product(pairs_a, pairs_b)]

        # list_works = np.array_split(matrix_node_indices, n_jobs)

        # matrix_node_indices = [(p[2], p[3],p[0][0], p[0][1], p[1][0], p[1][1]) for p in matrix_node_indices]
        # edge_propagation = parallel(delayed(
        # lambda list_indices: [propagation_func(p[2], p[3],
        #    p[0][0], p[0][1], p[1][0], p[1][1]) for p in list_indices])
        #         (list_indices) for list_indices in list_works)
        # lambda p: [propagation_func(p[2], p[3],
        #        p[0][0], p[0][1], p[1][0], p[1][1])])(p) for p in matrix_node_indices)
        # edge_propagation = [x for lst in edge_propagation for x in lst]
