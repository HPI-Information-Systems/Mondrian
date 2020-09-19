import multiprocessing
import os
import pickle
from pathlib import Path

import itertools
import random
from PIL import Image, ImageDraw
from joblib.parallel import Parallel, delayed, parallel_backend

from mondrian.distances import *
from mondrian.visualization import draw_rectangles

from sklearn.cluster import DBSCAN, MeanShift, OPTICS, cluster_optics_dbscan, SpectralClustering
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist


#Credits: https://gist.github.com/rtavenar/a4fb580ae235cc61ce8cf07878810567

def cdist_generic(dist_fun, dataset1, dataset2=None, n_jobs=None, verbose=0,
                  compute_diagonal=True, pool =None, *args, **kwargs):

    """Compute cross-similarity matrix with joblib parallelization for a given
    similarity function.
    Parameters
    ----------
    dist_fun : function
        Similarity function to be used. Should be a function such that
        `dist_fun(dataset1[i], dataset2[j])` returns a distance (a float).
    
    dataset1 : array-like
        A dataset
    
    dataset2 : array-like (default: None)
        Another dataset. If `None`, self-similarity of `dataset1` is returned.
    
    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See scikit-learns'
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n-jobs>`_
        for more details.
    
    verbose : int, optional (default=0)
        The verbosity level: if non zero, progress messages are printed.
        Above 50, the output is sent to stdout.
        The frequency of the messages increases with the verbosity level.
        If it more than 10, all iterations are reported.
        `Glossary <https://joblib.readthedocs.io/en/latest/parallel.html#parallel-reference-documentation>`_
        for more details.
    
    compute_diagonal : bool (default: True)
        Whether diagonal terms should be computed or assumed to be 0 in the
        self-similarity case. Used only if `dataset2` is `None`.
    
    *args and **kwargs :
        Optional additional parameters to be passed to the similarity function.
    
    Returns
    -------
    cdist : numpy.ndarray
        Cross-similarity matrix
    """ # noqa: E501
    if n_jobs is not None:
        if n_jobs ==-1:
            n_jobs = multiprocessing.cpu_count()

    if dataset2 is None:
        # Inspired from code by @GillesVandewiele:
        # https://github.com/rtavenar/tslearn/pull/128#discussion_r314978479
        matrix = np.zeros((len(dataset1), len(dataset1)))
        indices = np.triu_indices(len(dataset1),
                                     k=0 if compute_diagonal else 1,
                                     m=len(dataset1))
        matrix[indices] = Parallel(n_jobs=n_jobs,
                                   verbose=verbose)(
            delayed(dist_fun)(
                dataset1[i], dataset1[j],
                *args, **kwargs
            )
            for i in range(len(dataset1))
            for j in range(i if compute_diagonal else i + 1,
                           len(dataset1))
        )
        indices = np.tril_indices(len(dataset1), k=-1, m=len(dataset1))
        matrix[indices] = matrix.T[indices]
        return matrix

    else:
        index_pairs = list(itertools.product(range(len(dataset1)), range(len(dataset2))))
        if len(index_pairs) <128:
            n_jobs=1 #be nice
        list_works = [index_pairs]
        if n_jobs is not None:
            list_works = np.array_split(index_pairs, n_jobs)
        with parallel_backend("templates"):
            matrix = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(
                lambda list_pairs: [dist_fun(dataset1[p[0]], dataset2[p[1]],*args, **kwargs) for p in list_pairs])(list_pairs) for list_pairs in list_works)

        # matrix_old = Parallel(n_jobs=n_jobs, verbose=verbose)(
        #     delayed(dist_fun)(
        #         dataset1[i], dataset2[j],
        #         *args, **kwargs
        #     )
        #     for i in range(len(dataset1)) for j in range(len(dataset2))
        # )
        matrix = [x for lst in matrix for x in lst]
        return np.array(matrix).reshape((len(dataset1), -1))


# noinspection PyUnusedLocal
def clustering(elements, min_samples=1, print_stats=False, radius=1, alpha=1, beta=1, gamma=1, lbp=False,
               distances=None, n_jobs=-1, save_path= None):
    """
    Wrapper function to encapsulate the clustering process. It can use either a Meanshift, a DBSCAN with or without a custom metric.

    Parameters:
    -----------
    elements: sequence
        Elements to cluster e.g. a list of feature vectors.
    min_samples: int
        The minimum number of samples for each cluster, a parameter for DBSCAN
    save_path: str
        The path where the file should be saved to, including filename

    Returns:
    ---------
    labels: list
        For each element, the cluster label associated to it
    silhouette: float
        The silhouette score
    """

    save = False
    if save_path:
        try:
            labels = pickle.load(open(save_path, "rb"))
            return labels, 0
        except:
            save = True

    if len(elements) == 1:
        return [0], 0

    if distances is None:

        file_width = max([e["bot_rx"][0] for e in elements]) + 1
        file_height = max([e["bot_rx"][1] for e in elements]) + 1

        p_elements = np.asarray([rectangle_as_array(x) for x in elements])
        # distances = cdist_generic(dist_fun=parallel_distance, dataset1=p_elements, compute_diagonal=False, n_jobs = n_jobs,
        #                   file_width=file_width, file_height=file_height, alpha=alpha, beta=beta, gamma=gamma, lbp = lbp)

        distances = pairwise_distances(X=p_elements, metric=parallel_distance, n_jobs=n_jobs,
                                       file_width=file_width, file_height=file_height, alpha=alpha, beta=beta,
                                       gamma=gamma)


    db = DBSCAN(eps=radius, min_samples=min_samples, metric="precomputed", n_jobs=n_jobs)
    # db = SpectralClustering(n_clusters=eps, affinity="precomputed", n_jobs=n_jobs)
    db.fit(distances)

    labels = denoise_labels(db.labels_)
    n_clusters = len(set(labels))

    if print_stats:
        print('Estimated number of clusters: %d' % n_clusters)
        # print("Silhouette Coefficient: %0.3f" % silhouette)

    if -1 in labels:
        raise Exception("Clustering still found noise points ?!")

    if save:
        Path(os.path.split(save_path)[0]).mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(labels, f)

    return labels, 0


def describe_cluster(x, n_feat_clusters):
    y = []
    n_elements = np.shape(x)[0]
    unique = np.unique(x)
    n_unique = np.size(unique)
    y.append(n_elements)
    y.append(n_unique)
    y.append(n_unique / n_elements)
    for i in range(n_feat_clusters):
        if i in unique:
            y.append(1)
        else:
            y.append(0)
    return y


def find_cluster_edges(labels, elements):
    """
    This functions finds the corner points (upper-left and lower-right) of each cluster of elements found in an image.
    
    To find the most inclusive borders for each cluster, its corner points are updated whenever a new element 
    is found that spans out of the borders while still belonging to the same cluster.
    This means that the upper-left corner is always updated with the minimum coordinates values (coordinate-wise), 
    and the lower right with the maximum values (coordinate-wise).
    
    Parameters:
    -----------
    labels: list of integers
            For each element found, the cluster it belongs to. (As returned by scikit clustering objects)
            
    elements: list of dictionaries
            Each element is a rectangle represented as a dictionary
            Dictionary contains the keys "x","y" (the upper left corner coordinates), and "width","height".

    Returns:
    --------
    cluster_edges: list of dictionaries
        Each item of the list is a cluster, represented as a dictionary.
        Each dictionary contains the keys "top_lx" and "bot_rx", associated to a list of two integers [x,y].
    
    """

    tmp = {"top_lx": [float("inf"), float("inf")], "bot_rx": [-1, -1]}

    n_clusters = len(set(labels))
    cluster_edges = [tmp for _ in range(n_clusters)]

    for idx, e in enumerate(elements):
        cluster = labels[idx]
        current = cluster_edges[cluster]

        cluster_edges[cluster] = {"top_lx": np.minimum(current["top_lx"], e["top_lx"]), \
                                  "bot_rx": np.maximum(current["bot_rx"], e["bot_rx"])}

    return cluster_edges


def intersection_over_union(boxA, boxB, img):
    """
    This function computes the intersection over union given two rectangles.
    
    Parameters
    ----------
    boxA, boxB: bounding boxes
        Expressed as dictionaries with the keys "top_lx", "bot_rx", both having as a value a pair of integers.
    img: the content of the image

    Returns
    --------
    iou: float
        The value for the intersection over union for the two rectangles
    
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA["top_lx"][0], boxB["top_lx"][0])
    yA = max(boxA["top_lx"][1], boxB["top_lx"][1])
    xB = min(boxA["bot_rx"][0], boxB["bot_rx"][0])
    yB = min(boxA["bot_rx"][1], boxB["bot_rx"][1])

    # compute the area of intersection rectangle
    interArea = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)
    boxAArea = (boxA["bot_rx"][0] - boxA["top_lx"][0] + 1) * (boxA["bot_rx"][1] - boxA["top_lx"][1] + 1)
    boxBArea = (boxB["bot_rx"][0] - boxB["top_lx"][0] + 1) * (boxB["bot_rx"][1] - boxB["top_lx"][1] + 1)

    inter_region = Image.fromarray(img).crop((xA, yA, xB + 1, yB + 1))
    inter_region = np.asarray(inter_region)

    try: #inter_region can be empty
        # print(np.size(np.ravel(inter_region[:,:,0])))
        w, h = np.size(inter_region, 0), np.size(inter_region, 1)
        inter_n_empty = np.sum([1 for px in inter_region.reshape(w * h, 3) if np.array_equal(px, [255, 255, 255])])
        if inter_n_empty > 0:
            interArea = w * h - inter_n_empty
    except:
        pass

    # compute the area of both the prediction and ground-truth
    # rectangles

    A_region = Image.fromarray(img).crop(
        (boxA["top_lx"][0], boxA["top_lx"][1], boxA["bot_rx"][0] + 1, boxA["bot_rx"][1] + 1))
    A_region = np.asarray(A_region)

    B_region = Image.fromarray(img).crop(
        (boxB["top_lx"][0], boxB["top_lx"][1], boxB["bot_rx"][0] + 1, boxB["bot_rx"][1] + 1))
    B_region = np.asarray(B_region)

    A_w, A_h = np.size(A_region, 0), np.size(A_region, 1)
    A_n_empty = np.sum([1 for px in A_region.reshape(A_w * A_h, 3) if np.array_equal(px, [255, 255, 255])])
    if A_n_empty > 0:
        boxAArea = A_w * A_h - A_n_empty

    B_w, B_h = np.size(B_region, 0), np.size(B_region, 1)
    B_n_empty = np.sum([1 for px in B_region.reshape(B_w * B_h, 3) if np.array_equal(px, [255, 255, 255])])
    if B_n_empty > 0:
        boxBArea = B_w * B_h - B_n_empty

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def cluster_membership(element, target_clusters):
    """
    This function computes for a given element what cluster contains it, given cluster edges.
    
    Parameters:
        element: dictionary containing keys "top_lx" and "bot_rx"
        target_clusters: array of dictionaries containing keys "top_lx" and "bot_rx"
    
    Returns:
        idx: integer
            label of cluster, -1 if is not perfectly contained in any cluster
    
    """
    for idx, c in enumerate(target_clusters):
        e_x1 = element["top_lx"][0]
        e_x2 = element["bot_rx"][0]

        e_y1 = element["top_lx"][1]
        e_y2 = element["bot_rx"][1]

        corner_top = (e_x1 >= c["top_lx"][0]) and (e_x2 <= c["bot_rx"][0])
        corner_bot = (e_y1 >= c["top_lx"][1]) and (e_y2 <= c["bot_rx"][1])

        if corner_top and corner_bot:
            return idx

    return -1


def cluster_labels_inference(elements, clusters):
    """
    This function infers to what cluster any given element of a list belongs.
    
    Parameters:
        elements: array compatible with cluster_membership function
        clusters: array compatible with cluster_membership function
        
    Returns:
        cluster_labels: list
            The list of labels for the given elements
    
    """

    cluster_labels = [None for _ in range(np.size(elements))]

    for i, e in enumerate(elements):
        cluster_labels[i] = cluster_membership(e, clusters)

    return cluster_labels


def inverse_clustering(cluster_labels):
    n_clusters = max([0] + cluster_labels) + 1
    cluster_array = [[] for _ in range(n_clusters)]

    for idx, l in enumerate(cluster_labels):
        cluster_array[l].append(idx)

    return cluster_array


def denoise_labels(labels):
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    labels = list(labels)
    if n_noise > 0:
        for i in range(0, n_noise):
            labels[labels.index(-1)] = n_clusters + i
    return labels


def compute_edit_cost(elements, labels, target_edges):
    elements = [np.ravel(x) for x in elements]
    target_labels = cluster_labels_inference(elements, target_edges)
    target = inverse_clustering(target_labels)
    edit_cost = slice_algorithm(labels, target)
    return edit_cost


# user interactions
def compute_user_actions(iou_m, print_costs=False):
    add_cost = 2#3
    del_cost = 1#2
    resize_cost = 1

    total_cost = 0
    composition = ""
    n_add =0
    n_del = 0

    n_clusters = len(iou_m[0, :])
    n_tables = len(iou_m[:, 0])

    n_nonsparse_cols = n_clusters

    if n_tables > n_clusters:
        n_add += (n_tables - n_clusters)
    else:
        n_del += (n_clusters - n_tables)

    for j in range(n_clusters):
        if np.any(iou_m[:, j] == 1):
            n_nonsparse_cols -= 1
        if np.all(iou_m[:,j] == 0):
            n_del +=1
        # n_intersecting = len([x for x in iou_m[:,j] if x ==1])
        # if n_intersecting >1:
        #     print("N intersecting is ", n_intersecting)
        #     print("ioum", iou_m)
        #     n_nonsparse_cols += 1 #you need one resize
        #     total_cost = (n_intersecting-1) * add_cost #and n-1 additions

    for i in range(n_tables):
        if np.all(iou_m[i,:] == 0):
            total_cost += add_cost
            n_add +=1

    total_cost += n_add * add_cost
    total_cost += n_del * del_cost
    total_cost += n_nonsparse_cols * resize_cost
    composition += str(n_add) + " add, "
    composition += str(n_del) + "del, "
    composition += str(n_nonsparse_cols) + " resizes"

    if print_costs:
        print(iou_m)
        print("Total cost:", total_cost, " ", composition)
    return total_cost, composition


# Clusters must be expressed in terms of the elements that belong to them

def slice_algorithm(predicted_labels, target_clusters):
    """
    Algorithm to find the split-merge edit distance between two clustering over the same elements.
    How the algorithm works:
        For each cluster in the target, build a map that tells in what predicted cluster the elements
        of the target cluster lie.
        E.g.: Predicted {[0,1],[2,3]}, Target: {[0,1,2],[3]}
            For the first target cluster the map looks like {0:2, 1:1} meaning that two elements
            are in the first predicted cluster and one in the second.
            
        Then comes calculation of splits/merges operation. If the predicted cluster contains more elements
        that don't belong to the target cluster currently under inspection, there is the need for a split.
        If after the split the cluster is complete as it is, there is no need for a merge.
        Otherwise, each successive iteration will include a merge with the previous ones.
        
        Current split cost: 1 (regardless of # of elements split)
        Current merge cost: 1 (regardless of # of elements split)
    
    Parameters: 
        predicted_labels: list of integers
            For each element at index i, stores its predicted cluster label
        
        target_clusters: list of lists of integers
            Each cluster is represented with a list containing the index of the elements that belong to it.
            E.g. [[0,1],[2,3]]
            
    Returns:
        cost: integer
            Split-merge edit distance, range [0, inf)

    """

    _, r_sizes = np.unique(predicted_labels, return_counts=True)

    cost = 0

    for t in target_clusters:
        p_map = {}

        for element in t:
            if predicted_labels[element] not in p_map:
                p_map[predicted_labels[element]] = 0

            p_map[predicted_labels[element]] += 1

        # compute cost to generate cluster i

        c_cost = 0
        total_recs = 0

        for p in p_map:
            #         add the cost to split Ri
            count = p_map[p]
            if r_sizes[p] > count:
                c_cost += 1  # count #here goes cost function split

            r_sizes[p] -= count

            if total_recs != 0:
                c_cost += 1  # (total_recs) #here goes cost function merge

            total_recs += count

        cost += c_cost

    return cost


def evaluate_IoU(predicted_clusters, target_clusters, img, print_matrix=False, remove_col=False):
    # What happens in single case of file?

    iou_m = np.zeros((len(target_clusters), len(predicted_clusters)))

    tmp100 = 0
    tmp80 = 0
    tmp50 = 0

    dict_iou = {}

    for i, e in enumerate(target_clusters):
        max = 0
        for j, t in enumerate(predicted_clusters):
            score = intersection_over_union(e, t, img)
            iou_m[i, j] = score
            if score >= 1:
                tmp100 += 1
                tmp80 += 1
                tmp50 += 1
            elif score >= 0.8:
                tmp80 += 1
                tmp50 += 1
            elif score > 0.5:
                tmp50 += 1
            if score > max:
                max = score
        dict_iou[e["region_label"]] = max

    if remove_col:
        # TODO
        iou_m.argmax(axis=1)

    acc100 = tmp100 / len(target_clusters) * 100
    acc80 = tmp80 / len(target_clusters) * 100
    acc50 = tmp50 / len(target_clusters) * 100

    # if acc50 ==0:
    #     print_matrix = True

    if print_matrix:
        print(iou_m)

    return acc100, acc80, acc50, dict_iou, iou_m


def calculate_W(custom_metric_matrix, labels):
    W = 0
    for c in itertools.combinations(range(len(labels)), 2):
        i, j = c[0], c[1]
        if labels[i] == labels[j]:
            W += custom_metric_matrix[i, j]
    return W


def generate_random_elements(n, file_width, file_height):
    random_elements = []
    i = 0
    while i < n:
        x0 = random.randint(0, file_width)
        y0 = random.randint(0, file_height)
        x1 = random.randint(x0, file_width)
        y1 = random.randint(y0, file_height)

        redo = False
        for r in random_elements:
            if dist_closest(np.asarray([x0, y0, x1, y1]), rectangle_as_array(r)) == 0:
                redo = True
                break
        if redo:
            continue

        c_x = x0 + np.floor(file_width / 2)
        c_y = y0 + np.floor(file_height / 2)
        w = x1 - x0 + 1
        h = y1 - y0 + 1
        inner_area = np.random.randint(0, w * h)

        e = {
            "label": "random_" + str(i),
            "top_lx": [x0, y0],
            "bot_rx": [x1, y1],
            "center": [c_x, c_y],
            "aspect_ratio": float(w) / h,
            "width": w,
            "height": h,
            "area": w * h,
            "inner_area": inner_area,
            "extent": inner_area / (w * h),
            "diagonal": np.sqrt(file_width ** 2 + file_height ** 2)
        }
        random_elements.append(e)
        i += 1
    return random_elements


def optimal_gap(elements, alpha, beta, gamma, lbp, range_k, n_b=100):
    n_elements = len(elements)

    W_target = []
    W_random = []

    file_width = max([e["bot_rx"][0] for e in elements]) + 1
    file_height = max([e["bot_rx"][1] for e in elements]) + 1
    p_elements = np.asarray([rectangle_as_array(x) for x in elements])
    custom_metric_matrix = cdist(p_elements, p_elements,
                                 metric=lambda x, y, file_width=file_width, file_height=file_height, alpha=alpha,
                                               beta=beta, gamma=gamma, lbp=lbp:
                                 parallel_distance(x, y, file_width, file_height, alpha, beta, gamma, lbp))

    n_k = []
    radius_list = []
    latest_n_k = -1
    for k in range_k:
        db = DBSCAN(eps=k, min_samples=1, metric="precomputed", n_jobs=-1)
        # db = SpectralClustering(n_clusters = k, affinity = "precomputed")
        db.fit(custom_metric_matrix)
        labels = denoise_labels(db.labels_)
        cur_n_k = len(set(labels))
        if (cur_n_k > latest_n_k):
            latest_n_k = cur_n_k
            W_target.append(calculate_W(custom_metric_matrix, labels))
            n_k.append(len(set(labels)))
            radius_list.append(k)

    for b in range(n_b):
        r_elements = generate_random_elements(n_elements, file_width, file_height)

        if b == 0:
            img = Image.new("RGB", (file_width, file_height))
            FACTOR = 20  # Resize before otherwise lines are pixel wide
            img = img.resize(tuple(np.array(img.size) * FACTOR))
            draw = ImageDraw.Draw(img, "RGBA")
            draw_rectangles(draw, r_elements, factor=FACTOR)
            img.save('random.jpg')

        r_elements = np.asarray([rectangle_as_array(x) for x in r_elements])
        random_metric_matrix = cdist(r_elements, r_elements,
                                     metric=lambda x, y, file_width=file_width, file_height=file_height, alpha=alpha,
                                                   beta=beta, gamma=gamma, lbp=lbp:
                                     parallel_distance(x, y, file_width, file_height, alpha, beta, gamma, lbp))

        W_tmp = []
        for k in radius_list:
            db = DBSCAN(eps=k, min_samples=1, metric="precomputed", n_jobs=-1)
            # db = SpectralClustering(n_clusters=k, affinity="precomputed")
            db.fit(random_metric_matrix)
            labels = denoise_labels(db.labels_)
            W_tmp.append(calculate_W(random_metric_matrix, labels))
        W_random.append(W_tmp)

    W_random = np.log(np.array(W_random) + 1)  # np.finfo(float).eps)
    W_target = np.log(np.array(W_target) + 1)  # np.finfo(float).eps)

    s = np.std(W_random, 0) * np.sqrt(1 + 1 / n_b)
    W_random = np.median(W_random, 0)

    gap = []
    for idx, x in enumerate(radius_list):
        g = W_target[idx] - W_random[idx]
        gap.append(g)

    gap = np.abs(gap)
    #    gap = np.array(gap) + np.abs(min(gap))

    for i, x in enumerate(W_random):
        print("Radius:", radius_list[i], "N clusters", n_k[i], "W_target", W_target[i])
        print("\tW_random", W_random[i], "Gap", gap[i], "Sk", s[i])
        print("\t\t Gap - std", gap[i] - s[i])

    # optimal_k = radius_list[-1]
    # for i, g in enumerate(gap[:-1]):
    #     if gap[i] >= gap[i + 1]:  # - s[i + 1]:
    #         optimal_k = radius_list[i]  # biggest k such that gap(k)
    #         break

    optimal_k = radius_list[np.argmax(gap)]

    return optimal_k, radius_list, gap, W_target, W_random, s


def iou_labels(predicted_labels, target_labels):
    """ This function computes the average iou given clusters in the form of element labels
        IT IS SENSITIVE TO THE ARGUMENTORDER
    """
    iou, target_indices, predicted_indices = dict(), dict(), dict()


    for cluster in set(target_labels):
        target_indices[cluster] = set([idx for idx, t in enumerate(target_labels) if t==cluster])

    for cluster in set(predicted_labels):
        predicted_indices[cluster] = set([idx for idx, t in enumerate(predicted_labels) if t==cluster])

    for k in target_indices:
        max_iou = 0
        for pred_k,v in predicted_indices.items():
            intersection = len(v.intersection(target_indices[k]))
            if intersection >0:
                tmp = intersection/len(v.union(target_indices[k]))
                max_iou = tmp if tmp>max_iou else max_iou #taking the max
        iou[k] = max_iou

    avg = np.average([v for k,v in iou.items()], weights = [len(v) for k,v in target_indices.items()])
    #weighted
    return iou, avg
    # return np.average([v for k,v in iou.items()])
