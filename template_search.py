from __future__ import print_function

import json
import multiprocessing
import os
import pickle
import time
from pathlib import Path

import numpy as np
import networkx as nx
import pandas as pd

import builtins as __builtin__
import tqdm as tqdm
import warnings

from numpy import VisibleDeprecationWarning
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score, homogeneity_score, completeness_score, v_measure_score

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)
import argparse
from pathos.multiprocessing import ProcessingPool as Pool

from mondrian.model.spreadsheet import Spreadsheet
from mondrian.clustering import iou_labels
from mondrian.model.region import parallel_region_sim
from mondrian.parallel_similarities import parallel_layout_similarity
from mondrian.model.mondrian import calculate_layout

pd.options.mode.chained_assignment = None

DELIMITER = ","
OVERWRITE = "yes"
SUBSET_START = 0
SUBSET_END = None

result_dir = f"../multiregion-detection/results/"

to_skip = ["jane_tholt__13157__SRP No. 13.xlsx_Study 1 Delv Dth By Location.csv",
           "holden_salisbury__12317__UAMPS-Int_Ext.xlsx_UAMPS.csv",
           "253e3a7a-012e-4bb7-b5be-d458bb4fe936.xlsx_Data.csv"]


def print(*args, **kwargs):
    return __builtin__.print(f"\033[94m{time.process_time()}:\033[0m", *args, **kwargs)


def assess_pair(m, n, allthresholds = False):
    if not allthresholds:
        if min(m, n) / max(m, n) < 0.7:
            return "easy"
    if np.mean([m, n]) < 100:
        return "easy"
    else:
        return "hard"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="fuste", help="The dataset on which to perform experiments")
    parser.add_argument("--a", default=1, help="The desired alpha to experiment (default = 1)")
    parser.add_argument("--b", default=1, help="The desired beta to experiment (default = 1)")
    parser.add_argument("--g", default=1, help="The desired gamma to experiment (default = 1)")
    parser.add_argument("--p", default=True, help="1 for partitioning, 0 for no partitioning (default = 1)")
    parser.add_argument("--experiment", default="static", help="The experiment to pick clustering results from (default = 'static')")
    parser.add_argument("--r", default=None, help="The desired radius for a static radius experiment")
    parser.add_argument("--allthresholds", default=False, help="Do not skip computation for thresholds below 0.7 (default= False)")
    parser.add_argument("--rthreshold", default=0.75, help="The threshold for region similarity (default = 0.75)")
    parser.add_argument("--thresholdlist", nargs="*", type=float, default=list(np.arange(0.7, 1, 0.1)),
                        help="A list of thresholds for file similarity")
    parser.add_argument("--iteration", default=0, help="The iteration of the experiment for time calculation")
    parser.add_argument("--overwrite", default="", help="Whether or not to recalculate similarity pairs")

    args = parser.parse_args()
    alpha = float(args.a)
    beta = float(args.b)
    gamma = float(args.g)
    radius = float(args.r) if args.r is not None else None
    partitioning = bool(args.p)
    DATASET = args.dataset
    REGION_THRESHOLD = float(args.rthreshold)
    # TEMPLATE_THRESHOLD = float(args.tthreshold)
    thresholds_list = args.thresholdlist
    experiment = args.experiment
    allthresholds = args.allthresholds
    iteration = args.iteration
    overwrite = args.overwrite
    # dir = "res/files"
    file_dir = f"res/files/{DATASET}/"

    annotations_regions = f"res/{DATASET}/annotations/annotations_elements.json"
    target_regions = json.load(open(annotations_regions, "r"))
    region_files = [k for k in target_regions if k not in to_skip]

    annotations_templates = f"res/{DATASET}/annotations/annotations_templates.json"
    target_templates = json.load(open(annotations_templates, "r"))
    target_templates = {k: [f for f in v if f in region_files] for k, v in target_templates.items() if len([f for f in v if f in region_files]) != 0}

    hyperparameters = "a" + str(alpha) + "_b" + str(beta) + "_g" + str(gamma) + \
                      "_partition" + str(int(partitioning))

    run_path = os.path.join(result_dir, DATASET, hyperparameters, experiment)
    template_path = os.path.join("results/", DATASET, hyperparameters, experiment)
    if allthresholds:
        template_path+=("_allthresholds")
    Path(template_path).mkdir(parents=True, exist_ok=True)

    partition_path = f"res/json/{DATASET}/baseline/"

    global_time = time.time()
    print("Using results from:", run_path)
    if experiment == "static":
        run_path += f"/radius_{radius}/"
    if experiment == "baseline":
        run_path = os.path.join(result_dir, DATASET, "connected_components")

    total_cores = multiprocessing.cpu_count()
    n_cores = total_cores
    n_jobs = n_cores
    print("Total cores: ", total_cores, "used cores: ", n_cores)

    file_paths = []
    for index, file in enumerate(os.listdir(file_dir)[SUBSET_START:SUBSET_END]):
        fname = os.fsdecode(file)
        if not fname.endswith(".csv") or fname in to_skip or fname not in target_regions:
            continue
        fpath = os.path.join(file_dir, fname)
        file_paths.append(fpath)
    files_df = pd.DataFrame({"file_path": file_paths})

    print("Loading list of files layouts...")
    with Pool(n_cores) as pool:
        chunk_size = int(len(file_paths) / n_cores)
        args = zip(file_paths, [partition_path] * len(file_paths), [run_path] * len(file_paths), [overwrite] * len(file_paths))
        spreadsheet_list = list(pool.starmap(load_spreadsheet, args, chunksize = chunk_size))

    files_df["filename"] = [s.filename for s in spreadsheet_list]
    files_df["layout"] = [s.layout for s in spreadsheet_list]
    files_df["n_regions"] = [len(s.clusters) for s in spreadsheet_list]
    files_df["spreadsheet"] = spreadsheet_list
    files_df['index'] = files_df.index

    del spreadsheet_list, file_paths

    regions_df = pd.DataFrame([{"file": f, "region": r_idx, "color_hist": r.color_hist}
                               for f, s in zip(files_df["filename"].index, files_df["spreadsheet"]) for r_idx, r in enumerate(s.clusters)])

    regions_df["joinkey"] = 0
    regions_df['file_region'] = regions_df.index
    print("Creating region pairs - 1/2...")
    regions_df.set_index("joinkey", inplace=True)
    region_pairs_df = regions_df.join(regions_df, lsuffix="_x", rsuffix="_y", how="outer")
    region_pairs_df = region_pairs_df[region_pairs_df["file_x"] != region_pairs_df["file_y"]].reset_index(drop=True)
    region_pairs_df.astype(
        {"file_x": "int16", "file_y": "int16", "region_x": "int16", "region_y": "int16", "file_region_x": "int16", "file_region_y": "int16"},
        copy=False)
    print("Creating region pairs - 2/2...")
    region_pairs_df["pair"] = [frozenset(p) for p in zip(region_pairs_df["file_region_x"], region_pairs_df["file_region_y"])]

    print("Dropping duplicate pairs...")
    region_pairs_df = region_pairs_df.drop_duplicates(subset="pair", ignore_index=True)
    regions_df.set_index("file_region", inplace=True)
    region_pairs_df.drop(columns=["pair"], inplace=True)
    print(f"There are {len(regions_df)} total regions")
    print("Number of pairs", len(region_pairs_df))

    region_sims_pckl = f"{template_path}/region_similarities.pckl"
    try:
        diff = region_pairs_df
        start = time.time()
        with open(region_sims_pckl+overwrite, "rb") as pckl_file:
            loaded_df = pickle.load(pckl_file)
        print(f"Time for loading region similarities: {time.time() - start} sec")

        region_pairs_df = region_pairs_df.merge(loaded_df[["file_region_x", "file_region_y", "similarity"]], how="left",
                                                on=["file_region_x", "file_region_y"])
        diff = region_pairs_df[np.isnan(region_pairs_df["similarity"])]
        diff = diff[["file_region_x", "file_region_y", "color_hist_x", "color_hist_y", "similarity"]]
        del loaded_df
        assert len(diff) == 0
    except (FileNotFoundError, EOFError, AssertionError) as e:
        print("Computing", len(diff), " missing region similarities")
        start = time.time()
        hist_pairs = list(zip(diff["color_hist_x"], diff["color_hist_y"]))
        chunksize = int(len(hist_pairs) / n_cores)

        with multiprocessing.pool.Pool(n_cores) as pool:
            scores = list(pool.starmap(parallel_region_sim, hist_pairs, chunksize=chunksize))

        if "similarity" in region_pairs_df.columns:
            diff["similarity"] = scores
            region_pairs_df.update(diff["similarity"])
        else:
            region_pairs_df["similarity"] = scores

        del scores
        print(f"Time for computing region similiraties: {time.time() - start} sec")
        with open(region_sims_pckl, "wb") as pckl_file:
            pickle.dump(region_pairs_df, pckl_file)
    del diff

    print(f"Finding candidate file pairs... {time.time() - start} sec")
    layout_pairs_df = region_pairs_df[region_pairs_df["similarity"] > REGION_THRESHOLD]
    layout_pairs_df = layout_pairs_df[["file_x", "file_y"]]
    layout_pairs_df["pair"] = [frozenset(p) for p in zip(layout_pairs_df["file_x"], layout_pairs_df["file_y"])]
    print("Dropping duplicate pairs...")
    layout_pairs_df = layout_pairs_df.drop_duplicates(subset="pair", ignore_index=True)
    layout_pairs_df.drop(columns="pair", inplace=True)

    layout_pairs_df = layout_pairs_df.merge(files_df[["n_regions"]], how="left", left_on=["file_x"], right_index=True)
    layout_pairs_df.rename(columns={"n_regions": "n_regions_x"})
    layout_pairs_df = layout_pairs_df.merge(files_df[["n_regions"]], how="left", left_on=["file_y"], right_index=True)
    layout_pairs_df.rename(columns={"n_regions": "n_regions_y"})

    print("There are", len(layout_pairs_df), "candidate layout pairs")
    print("Assessing pair difficulty...")

    chunksize = int(len(layout_pairs_df) / n_cores)
    with multiprocessing.pool.Pool(n_cores) as pool:
        difficulty = list(pool.starmap(assess_pair, zip(layout_pairs_df["n_regions_x"], layout_pairs_df["n_regions_y"], [allthresholds]*len(layout_pairs_df)), chunksize=chunksize))
    layout_pairs_df["difficulty"] = difficulty

    easy_pairs_df = layout_pairs_df[layout_pairs_df["difficulty"] == "easy"].reset_index(drop=True).copy()
    hard_pairs_df = layout_pairs_df[layout_pairs_df["difficulty"] == "hard"].reset_index(drop=True).copy()
    easy_pairs_df.drop(columns="difficulty", inplace=True)
    hard_pairs_df.drop(columns="difficulty", inplace=True)
    del layout_pairs_df, region_pairs_df, regions_df

    print(f"Number of easy pairs {len(easy_pairs_df)}, number of hard pairs {len(hard_pairs_df)}")

    easy_pairs_df.to_json(f"{template_path}/easy_pairs.json")
    hard_pairs_df.to_json(f"{template_path}/hard_pairs.json")
    easy_sims_path = f"{template_path}/easy_similarities.json"
    hard_sims_path = f"{template_path}/hard_similarities.json"
    try:
        easy_diff = easy_pairs_df
        start = time.time()
        if not os.path.exists(easy_sims_path+overwrite):
            raise FileNotFoundError(f"No easy similarities found in {easy_sims_path}")
        loaded_sims = pd.read_json(easy_sims_path)
        print(f"Loaded", len(loaded_sims), f"easy similarities in {time.time() - start} sec")
        assert len(loaded_sims) == len(easy_pairs_df)  # You only live once. Easy diff took too much time
        easy_pairs_df = loaded_sims
    except (FileNotFoundError, AssertionError) as e:
        print(e)
        print("Computing", len(easy_diff), "missing easy similarities")
        start = time.time()

        easy_diff = easy_diff.merge(files_df[["layout"]], how="left", left_on=["file_x"], right_index=True)
        easy_diff.rename(columns={"layout": "layout_x"})
        easy_diff = easy_diff.merge(files_df[["layout"]], how="left", left_on=["file_y"], right_index=True)
        easy_diff.rename(columns={"layout": "layout_y"})

        args = list(zip(easy_diff["layout_x"], easy_diff["layout_y"], [allthresholds]*len(easy_diff)))
        print(f"Splitting workload into {n_cores} tasks of approximately {int(len(args) / n_jobs)} easy pairs each")
        with multiprocessing.pool.Pool(n_jobs) as pool:
            scores = list(pool.starmap(parallel_layout_similarity, args, chunksize=int(len(args) / n_jobs)))

        if "similarity" in easy_pairs_df.columns:
            easy_diff["similarity"] = scores
            easy_pairs_df.update(easy_diff["similarity"])
        else:
            easy_pairs_df["similarity"] = scores

        print(f"Time for computing easy similiraties: {time.time() - start} sec")
        if SUBSET_END is None:
            easy_pairs_df.to_json(easy_sims_path)
        del easy_diff
    try:
        del loaded_sims
    except:
        pass

    if len(hard_pairs_df) > 0:
        try:
            hard_diff = hard_pairs_df
            start = time.time()
            if not os.path.exists(hard_sims_path+overwrite):
                raise FileNotFoundError
            loaded_sims = pd.read_json(hard_sims_path)
            print(f"Loaded", len(loaded_sims), f"hard similarities in {time.time() - start} sec")
            print("Setting indices...")
            # Since these are harder pairs we try to minimize
            loaded_sims["idx"] = loaded_sims["file_x"] + loaded_sims["file_y"]
            hard_pairs_df["idx"] = hard_pairs_df["file_x"] + hard_pairs_df["file_y"]
            loaded_sims.set_index(["idx"], inplace=True)
            hard_pairs_df.set_index(["idx"], inplace=True)
            print("Joining dataframes...")
            hard_pairs_df = hard_pairs_df.join(loaded_sims[["similarity"]], how="left")
            print("After join")
            hard_diff = hard_pairs_df[np.isnan(hard_pairs_df["similarity"])]
            del loaded_sims
            assert len(hard_diff) == 0
        except (FileNotFoundError, AssertionError) as e:
            start = time.time()

            print("Zipping hard layouts...")
            hard_layouts = zip(hard_diff["file_x"], hard_diff["file_y"])
            hard_layouts = [(files_df.loc[f1].spreadsheet.layout,
                             files_df.loc[f2].spreadsheet.layout) for f1, f2 in hard_layouts]

            print("Computing", len(hard_diff), "missing hard similarities")
            scores = list(tqdm.tqdm(map(lambda p: parallel_layout_similarity(p[0], p[1], n_jobs=n_cores, allthresholds=allthresholds), hard_layouts)))

            if "similarity" in hard_pairs_df.columns:
                hard_diff["similarity"] = scores
                hard_pairs_df.update["similarity"] = scores
            else:
                hard_pairs_df["similarity"] = scores
            print(f"Time for computing hard similiraties: {time.time() - start} sec")
            hard_pairs_df.to_json(hard_sims_path)

    print("Concatenating dataframes...")
    easy_pairs_df.reset_index(drop=True, inplace=True)
    hard_pairs_df.reset_index(drop=True, inplace=True)
    layout_pairs_df = pd.concat([easy_pairs_df, hard_pairs_df])

    print(f"Time for layout similiraties: {time.time() - start} sec")

    print("Calculating clusters of file templates")
    print("Threshold list", thresholds_list)

    with Pool(n_cores) as pool:
        chunk_size = int(len(thresholds_list)/n_cores)
        args = list(zip([layout_pairs_df] * len(thresholds_list), thresholds_list, [files_df] * len(thresholds_list), [template_path] * len(thresholds_list),
                [region_files] * len(thresholds_list), [target_templates] * len(thresholds_list)))
        result_list = list(pool.starmap(calculate_clusters, args, chunksize=chunk_size))

    print(result_list)

    print(f"Took {time.time() - start} seconds")
    with open(template_path + "/computation_time_"+str(iteration)+".txt", "w") as f:
        f.write("Iteration "+ str(iteration) + "took " + str(time.time() - global_time) + "seconds.")
    print("Done.")


def calculate_clusters(layout_pairs_df, threshold, files_df, template_path, region_files, target_templates):
    pickle.dump("test", open("./test.pckl", "wb"))

    interesting_pairs_df = layout_pairs_df[layout_pairs_df["similarity"] > threshold]
    interesting_pairs_df = interesting_pairs_df.merge(files_df[["filename"]], how="left", left_on=["file_x"], right_index=True)
    interesting_pairs_df.rename(columns={"filename": "filename_x"})
    interesting_pairs_df = interesting_pairs_df.merge(files_df[["filename"]], how="left", left_on=["file_y"], right_index=True)
    interesting_pairs_df.rename(columns={"filename": "filename_y"})

    threshold = round(threshold, 2)

    G = nx.Graph()
    G.add_nodes_from(files_df.filename.values)
    pairs = zip(interesting_pairs_df.filename_x.values, interesting_pairs_df.filename_y.values)
    G.add_edges_from(pairs)
    predicted_templates = nx.connected_components(G)
    predicted_templates = {"template_" + str(idx): list(t) for idx, t in enumerate(predicted_templates)}

    if SUBSET_END is not None:
        for k, s in predicted_templates.items():
            print(f"{k}: ")
            print("\t" + "\n".join(map(str, s)))
        raise DatasetNotCompleteError

    threshold_path = template_path + "/thresholds"
    Path(threshold_path).mkdir(parents=True, exist_ok=True)
    with open(threshold_path + "/predicted_templates_threshold" + str(threshold) + ".json", "w") as f:
        json.dump({k: v for k, v in predicted_templates.items()}, f)

    predicted_labels = []
    for f in region_files:
        for idx, k in enumerate(predicted_templates):
            if f in predicted_templates[k]:
                predicted_labels.append(idx)
                break

    target_labels = []
    for f in region_files:
        for idx, k in enumerate(target_templates):
            if f in target_templates[k]:
                target_labels.append(idx)
                break

    template_scores, avg_iou = iou_labels(predicted_labels, target_labels)
    nmi = normalized_mutual_info_score(predicted_labels, target_labels)
    ami = adjusted_mutual_info_score(predicted_labels, target_labels)
    homo = homogeneity_score(target_labels, predicted_labels)
    completeness = completeness_score(target_labels, predicted_labels)
    vm = v_measure_score(target_labels, predicted_labels)
    print("Template threshold:", threshold)
    print("Normalized mutual info:", nmi)
    print("Adjusted mutual info:", ami)
    print("Homogeneity score:", homo)
    print("Completeness score:", completeness)
    print("V measure score:", vm)
    print("Average weighted iou", avg_iou)

    pickle.dump(template_scores, open(template_path + "/template_scores.pckl", "wb"))
    json.dump(template_scores, open(template_path + "/template_scores.json", "w"))
    return 0

def load_spreadsheet(file_path, partition_path, run_path, overwrite = ""):
    s = Spreadsheet(file_path, printing=False, save_path=partition_path)
    clusters_path = f"{run_path}/{s.filename}_results.pckl"
    layout_path = f"{run_path}/{s.filename}_layout.pckl"

    s.clusters = s.restore(clusters_path, pckl=True)
    s.layout = calculate_layout(s.clusters, save_path=layout_path, overwrite=overwrite)
    return s

if __name__ == "__main__":
    main()
