# This script calculates the regions for each file, given a set of hyperparameters
# The input is a set of files in the /res/{dataset} folder, with their annotations in /res/{dataset}/annotations
# Saves the results as a pckl file and possibly the radii in a csv
# Launch with --help to see command line parameters

import argparse
import csv
import json, os
import math
import multiprocessing
import time
from functools import reduce
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.metrics import pairwise_distances

from mondrian.clustering import clustering, cluster_labels_inference, find_cluster_edges, iou_labels, evaluate_IoU
from mondrian.distances import rectangle_as_array, parallel_distance
from mondrian.visualization import table_as_image, find_table_elements

import numpy as np
import pickle

to_skip = ["jane_tholt__13157__SRP No. 13.xlsx_Study 1 Delv Dth By Location.csv",
           "holden_salisbury__12317__UAMPS-Int_Ext.xlsx_UAMPS.csv",
           "253e3a7a-012e-4bb7-b5be-d458bb4fe936.xlsx_Data.csv"]

to_include = ["cara_semperger__1463__February_Everything_Master.xlsx_Active Counterparties Jan 1.csv",
              "5e66b83d-3e77-446a-982f-edab5f6a447d.xlsx_Data.csv"]

total_cores = multiprocessing.cpu_count()
n_cores = total_cores
print("Total cores: ", total_cores, "used cores: ", n_cores)

parser = argparse.ArgumentParser()
parser.add_argument("--a", default=1, help="The desired alpha to experiment (default = 1)")
parser.add_argument("--b", default=1, help="The desired beta to experiment (default = 1)")
parser.add_argument("--g", default=1, help="The desired gamma to experiment (default = 1)")
parser.add_argument("--p", default=True, help="1 for partitioning, 0 for no partitioning (default = 1)")
parser.add_argument("--bestradius", default=False, action='store_true', help="To find the best radius")
parser.add_argument("--static", default=0, help="To manually use a static radius (default = 0)")
parser.add_argument("--subset", default=None, help="To just use a data subset")
parser.add_argument("--baseline", default=False, action='store_true', help="To select the baseline (default = False)")
parser.add_argument("--evaluate", default=False, action='store_true', help="To save the evaluation results (default = False)")
parser.add_argument("--experiment", default="static", help="The experiment in case it's evaluation only (default = 'static')")
parser.add_argument("--iteration", default="", help="The iteration in case it's evaluation only")
parser.add_argument("--dataset", default="fuste", help="The dataset on which to perform experiments (default = 'fuste')")

args = parser.parse_args()
alpha = float(args.a)
beta = float(args.b)
gamma = float(args.g)
partitioning = bool(args.p)
dataset = args.dataset
subset = int(args.subset) if args.subset is not None else None
csv_dir = "./res/" + dataset + "/csv/"
pckl_dir = "./res/" + dataset + "/pckl/"

hyperparameters = "a" + str(alpha) + "_b" + str(beta) + "_g" + str(gamma) + \
                  "_partition" + str(int(partitioning))

with open(f'res/{dataset}/annotations/annotations_elements.json') as json_file:
    data = json.load(json_file)


def main():
    if args.bestradius:
        experiment = "best-radius"
        result_dir = os.path.join("./results/", dataset, hyperparameters, experiment)
        if subset is not None:
            result_dir += "subset" + str(subset)
        Path(result_dir).mkdir(parents=True, exist_ok=True)

        evaluations = find_radii(result_dir)

    if float(args.static) > 0:
        experiment = "static"
        mult = float(args.static)
        mult = np.round(mult, 2)
        iteration = "radius_" + str(mult)
        result_dir = os.path.join("./", "results/", dataset, hyperparameters, experiment, iteration)
        if subset is not None:
            result_dir += "subset" + str(subset)
        Path(result_dir).mkdir(parents=True, exist_ok=True)

        evaluations = static_radii(mult, result_dir)

    if args.baseline:
        result_dir = os.path.join("./", "results/", dataset, "connected_components")
        if subset is not None:
            result_dir += "subset" + str(subset)
        Path(result_dir).mkdir(parents=True, exist_ok=True)

        evaluations = baseline_results(result_dir)

    if args.evaluate:
        try:
            if evaluations:
                print("Evaluation phase")
            dict_label_iou = [x[3] for x in evaluations]
            dict_label_iou = reduce(lambda a, b: dict(a, **b), dict_label_iou)
            label_iou_50 = len([x for x in dict_label_iou if dict_label_iou[x] > 0.5]) / len(dict_label_iou)
            label_iou_80 = len([x for x in dict_label_iou if dict_label_iou[x] > 0.8]) / len(dict_label_iou)
            label_iou_100 = len([x for x in dict_label_iou if dict_label_iou[x] == 1]) / len(dict_label_iou)

            label_iou_path = os.path.join(result_dir, "label_iou_scores.pckl")
            pickle.dump(dict_label_iou, open(label_iou_path, "wb"))
            print("% NEW! Clusters with IoU >50", label_iou_50)
            print("% NEW! Clusters with IoU >80", label_iou_80)
            print("% NEW! Clusters with IoU >100", label_iou_100)

        except:
            experiment = args.experiment
            iteration = args.iteration

            if experiment == "baseline":
                result_dir = os.path.join("./", "results/", dataset, "connected_components")
            elif "elvis" in experiment:
                result_dir = os.path.join("./", "results/", dataset, experiment, iteration)
            else:
                result_dir = os.path.join("./", "results/", dataset, hyperparameters, experiment, iteration)
            if subset is not None:
                result_dir += "subset" + str(subset)

            evaluations = Parallel(n_jobs=n_cores)(delayed(evaluate_file)(f, csv_dir, result_dir)
                                                   for f in list(data)[:subset] if f not in to_skip)
            print("Finished having evaluations")

        eval_time = time.time()
        dict_region_iou = [x[0] for x in evaluations]
        dict_binary = [x[1] for x in evaluations]

        acc_100 = np.average(
            [len([dict[region] for region in dict if dict[region] >= 1]) /
             len([dict[region] for region in dict])
             for dict in dict_region_iou])

        acc_80 = np.average(
            [len([dict[region] for region in dict if dict[region] >= 0.8]) /
             len([dict[region] for region in dict])
             for dict in dict_region_iou])

        acc_50 = np.average(
            [len([dict[region] for region in dict if dict[region] >= 0.5]) /
             len([dict[region] for region in dict])
             for dict in dict_region_iou])

        dict_region_iou = reduce(lambda a, b: dict(a, **b), dict_region_iou)
        dict_binary = reduce(lambda a, b: dict(a, **b), dict_binary)

        tn = len([x for x in dict_binary if dict_binary[x] == "tn"])
        tp = len([x for x in dict_binary if dict_binary[x] == "tp"])
        fn = len([x for x in dict_binary if dict_binary[x] == "fn"])
        fp = len([x for x in dict_binary if dict_binary[x] == "fp"])

        print("tn =", tn, "fn = ", fn)
        print("fp =", fp, "tp = ", tp)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        try:
            precision = tp / (tp + fp)
        except ZeroDivisionError:
            precision = 0
        try:
            recall = tp / (tp + fn)
        except ZeroDivisionError:
            recall = 0
        try:
            f1 = 2 * tp / (2 * tp + fp + fn)
        except ZeroDivisionError:
            f1 = 0

        iou_50 = len([x for x in dict_region_iou if dict_region_iou[x] > 0.5]) / len(dict_region_iou)
        iou_80 = len([x for x in dict_region_iou if dict_region_iou[x] > 0.8]) / len(dict_region_iou)
        iou_100 = len([x for x in dict_region_iou if dict_region_iou[x] == 1]) / len(dict_region_iou)

        iou_path = os.path.join(result_dir, "iou_scores.pckl")
        pickle.dump(dict_region_iou, open(iou_path, "wb"))

        binary_path = os.path.join(result_dir, "binary_scores.pckl")
        pickle.dump(dict_binary, open(binary_path, "wb"))

        print("Average accuracy at 50%  IoU: ", acc_50)
        print("Average accuracy at 80%  IoU: ", acc_80)
        print("Average accuracy at 100% IoU: ", acc_100)
        print("")
        print("Binary accuracy: ", accuracy)
        print("Binary f1: ", f1)
        print("Binary precision: ", precision)
        print("Binary recall: ", recall, "\n")
        print("")
        print("% Regions with IoU >50", iou_50)
        print("% Regions with IoU >80", iou_80)
        print("% Regions with IoU >100", iou_100)
        print("\nTime for evaluation", time.time() - eval_time)

    print("Done.")


def baseline_results(result_dir):
    execution_time = time.time()

    print("Experimenting with the baseline")

    evaluations = Parallel(n_jobs=n_cores)(delayed(process_file)(
        file, result_dir
    ) for file in data if file not in to_skip)

    print("\nOverall execution time", time.time() - execution_time)

    return evaluations


def static_radii(mult, result_dir):
    execution_time = time.time()

    print("Partitioning", partitioning, "Alpha", alpha, "Beta", beta, "Gamma ", gamma)
    print("\tRadius ", mult)

    evaluations = Parallel(n_jobs=n_cores)(delayed(process_file)(
        filename=file, result_dir=result_dir, radius=mult
    ) for file in list(data)[:subset] if file not in to_skip)

    print("\nOverall execution time", time.time() - execution_time)

    return evaluations


def find_radii(result_dir):
    csv_path = os.path.join(result_dir, "optimal_radii.csv")
    print("Writing first row")
    with open(csv_path, 'w') as fd:
        writer = csv.writer(fd, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["filename", "optimal_radius", "max_optimal_radius"])

    execution_time = time.time()

    print("Partitioning", partitioning, "Alpha", alpha, "Beta", beta, "Gamma ", gamma)

    evaluations = []

    for file in list(data)[:subset]:
        if file not in to_skip:
            evaluations.append(process_file(file, result_dir))

    print("\nOverall execution time", time.time() - execution_time)

    return evaluations


def process_file(filename, result_dir, radius=None):
    path = csv_dir + filename

    img = table_as_image(path)
    sample = data[filename]
    return_label_iou = {}
    return_dict_iou = {}
    return_dict_binary = {}

    if partitioning and not args.baseline:
        try:
            elements_found = pickle.load(open(pckl_dir + filename + "_partitions.pckl", "rb"))
        except:
            elements_found, _, _ = find_table_elements(img, partitioning)
            pickle.dump(elements_found, open(pckl_dir + filename + "_partitions.pckl", "wb"))
    else:
        elements_found, _, _ = find_table_elements(img, partitioning=False)

    if args.baseline:
        predicted_labels = [idx for idx, e in enumerate(elements_found)]
        list_radii = [evaluate_clustering_iteration(img, elements_found, predicted_labels, radius, sample, exec_time=0)]

    elif not args.baseline:
        file_width = max([e["bot_rx"][0] for e in elements_found]) + 1
        file_height = max([e["bot_rx"][1] for e in elements_found]) + 1

        try:
            distance_path = os.path.join("./results/", dataset, hyperparameters, "distances", filename + "_distances.pckl")
            distances = pickle.load(open(distance_path, "rb"))
        except:
            p_elements = np.asarray([rectangle_as_array(x) for x in elements_found])

            distances = pairwise_distances(X=p_elements, metric=parallel_distance, n_jobs=n_cores,
                                           file_width=file_width, file_height=file_height, alpha=alpha, beta=beta,
                                           gamma=gamma)
            distance_path = os.path.join("./results/", dataset, hyperparameters, "distances", filename + "_distances.pckl")
            Path(os.path.split(distance_path)[0]).mkdir(parents=True, exist_ok=True)
            pickle.dump(distances, open(distance_path, "wb"))
            print("Dumped distances into ", distance_path)

        if radius is not None:
            exec_time = time.time()
            predicted_labels, _ = clustering(elements_found, distances=distances, radius=radius,
                                             alpha=alpha,
                                             beta=beta, gamma=gamma,
                                             n_jobs=n_cores)
            exec_time -= time.time()
            list_radii = [evaluate_clustering_iteration(img, elements_found, predicted_labels, radius, sample, exec_time)]

        elif args.bestradius:
            iterations = np.arange(0.1, 2.1, 0.1)
            iterations = np.append(iterations, np.arange(3, 11, 1))
            iterations = np.append(iterations, np.arange(20, 110, 10))
            exec_time = time.time()

            list_prediction = Parallel(n_jobs=n_cores)(delayed(clustering)(
                elements_found, distances=distances, radius=mult,
                alpha=alpha, beta=beta, gamma=gamma, n_jobs=n_cores) for mult in iterations)

            exec_time = (time.time() - exec_time) / len(iterations)  # that is an ESTIMATE

            list_radii = Parallel(n_jobs=n_cores)(delayed(evaluate_clustering_iteration)(
                img, elements_found, clustering_results[0], iterations[idx], sample, exec_time)
                                                  for idx, clustering_results in enumerate(list_prediction))

        else:
            raise BaseException("No radius nor bestradius specified")

    best_overall = -1
    largest_radius = -1
    for dict in list_radii:
        overall = dict["overall"]

        if overall > best_overall:
            best_label_iou = dict["dict_label_iou"]
            best_overall = overall
            best_radius = dict["radius"]
            best_predicted_labels = dict["predicted_labels"]
            best_predicted_multiregion = dict["predicted_multiregion"]
            best_dict_iou = dict["dict_iou"]
            best_target_multiregion = dict["target_multiregion"]
            best_predicted_cluster_edges = dict["predicted_cluster_edges"]
            best_time = dict["time"]
            if best_radius is not None and largest_radius < best_radius:
                largest_radius = best_radius
        elif overall == best_overall:
            largest_radius = dict["radius"]

    for region in best_dict_iou:
        return_dict_iou[filename + "_" + region] = best_dict_iou[region]
    for region in best_label_iou:
        return_label_iou[filename + "_" + region] = best_label_iou[region]

    if best_predicted_multiregion and best_target_multiregion:
        return_dict_binary[filename] = "tp"
    elif best_predicted_multiregion and not best_target_multiregion:
        return_dict_binary[filename] = "fp"
    elif not best_predicted_multiregion and not best_target_multiregion:
        return_dict_binary[filename] = "tn"
    elif not best_predicted_multiregion and best_target_multiregion:
        return_dict_binary[filename] = "fn"

    target_cluster_edges = sample["regions"]
    target_labels = cluster_labels_inference(elements_found, target_cluster_edges)

    file_path = os.path.join(result_dir, filename + "_results.pckl")
    file_path_labels = os.path.join(result_dir, filename + "_results_labels.pckl")
    Path(os.path.split(file_path)[0]).mkdir(parents=True, exist_ok=True)
    pickle.dump([best_predicted_cluster_edges, target_cluster_edges, best_time], open(file_path, "wb"))
    pickle.dump([best_predicted_labels, target_labels], open(file_path_labels, "wb"))

    if args.bestradius:
        csv_path = os.path.join(result_dir, "optimal_radii.csv")
        with open(csv_path, 'a') as fd:
            writer = csv.writer(fd, quoting=csv.QUOTE_MINIMAL)
            writer.writerow([filename, best_radius, largest_radius])

    return return_dict_iou, return_dict_binary, return_label_iou


def evaluate_clustering_iteration(img, elements_found, predicted_labels, radius_eval, sample, exec_time):
    predicted_n_clusters = len(set(predicted_labels))
    predicted_cluster_edges = find_cluster_edges(predicted_labels, elements_found)

    target_cluster_edges = sample["regions"]
    target_n_clusters = sample["n_regions"]
    target_labels = cluster_labels_inference(elements_found, target_cluster_edges)

    dict_label_iou, avg_label_iou = iou_labels(predicted_labels=predicted_labels, target_labels=target_labels)
    dict_label_iou = {sample["regions"][k]["region_label"]: v for k, v in dict_label_iou.items()}

    acc_100, acc_80, acc_50, dict_iou, iou_m = evaluate_IoU(predicted_cluster_edges, target_cluster_edges, img)

    overall = np.average([dict_iou[x] for x in dict_iou])
    predicted_multiregion = (predicted_n_clusters > 1)
    target_multiregion = (target_n_clusters > 1)

    dict_mult = {"radius": radius_eval, "overall": overall, "predicted_labels": predicted_labels,
                 "target_labels": target_labels,  # TODO delete
                 "predicted_multiregion": predicted_multiregion, "target_multiregion": target_multiregion,
                 "predicted_cluster_edges": predicted_cluster_edges,
                 "dict_label_iou": dict_label_iou, "avg_label_iou": avg_label_iou,
                 "dict_iou": dict_iou, "time": exec_time}

    return dict_mult


def evaluate_file(filename, csv_dir, result_dir):
    result_file = filename + "_results.pckl"
    dict_regions_iou, dict_binary = {}, {}

    predicted_cluster_edges, target_cluster_edges, execution_time = pickle.load(
        open(os.path.join(result_dir, result_file), "rb"))
    predicted_n_clusters = len(predicted_cluster_edges)
    target_n_clusters = len(target_cluster_edges)

    file_path = csv_dir + filename
    img = table_as_image(file_path, color=False)
    acc_100, acc_80, acc_50, dict_iou, iou_m = evaluate_IoU(predicted_cluster_edges, target_cluster_edges, img)

    predicted_multiregion = (predicted_n_clusters > 1)
    target_multiregion = (target_n_clusters > 1)

    if predicted_multiregion and target_multiregion:
        dict_binary[filename] = "tp"
    elif predicted_multiregion and not target_multiregion:
        dict_binary[filename] = "fp"
    elif not predicted_multiregion and not target_multiregion:
        dict_binary[filename] = "tn"
    elif not predicted_multiregion and target_multiregion:
        dict_binary[filename] = "fn"

    for region in dict_iou:
        dict_regions_iou[filename + "_" + region] = dict_iou[region]

    return [dict_regions_iou, dict_binary]


if __name__ == "__main__":
    main()
