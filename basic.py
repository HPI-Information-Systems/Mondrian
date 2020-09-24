"""
This script can be used to test Mondrian on two csv files.
Using command line arguments, it is possible to edit the parameters for the clustering approach,
 as well as visualize and save the files with the detected regions as .png images.
"""

import argparse
import multiprocessing
import matplotlib.pyplot as plt

from mondrian.model.region import Region
from mondrian.clustering import clustering
from mondrian.model.mondrian import calculate_layout
from mondrian.visualization import table_as_image, find_table_elements

from mondrian.clustering import find_cluster_edges
from mondrian.model.spreadsheet import Spreadsheet
from mondrian.parallel_similarities import parallel_layout_similarity

N_CORES = multiprocessing.cpu_count()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fileA", default = "", help="The path of the first file to compare")
    parser.add_argument("--fileB", default= "", help="The path of the second file to compare")
    parser.add_argument("--a", default=1, help="The desired alpha to experiment")
    parser.add_argument("--b", default=0.5, help="The desired beta to experiment")
    parser.add_argument("--g", default=1, help="The desired gamma to experiment")
    parser.add_argument("--radius", default=1.4, help="The desired radius to experiment")
    parser.add_argument("--delimiter", default=",", help="The delimiter of the csv files")
    parser.add_argument("--printimages", default=False, help="Whether to print file images or not")

    args = parser.parse_args()
    f_a = args.fileA
    if f_a == "":
        f_a = input("Path of first file:")
    f_b = args.fileB
    if f_b == "":
        f_b = input("Path of second file:")
    alpha = float(args.a)
    beta = float(args.b)
    gamma = float(args.g)
    radius = float(args.radius)
    delimiter = args.delimiter
    printimages = bool(args.printimages)

    img_a = table_as_image(f_a, delimiter, color=True)
    img_b = table_as_image(f_b, delimiter, color=True)

    elements_a, _, _ = find_table_elements(img_a, partitioning=True)
    clusters_a, _ = clustering(elements_a, n_jobs=N_CORES, alpha=alpha, beta=beta, gamma=gamma, radius=radius)
    regions_a = [Region(f_a,img=img_a, **r) for r in find_cluster_edges(clusters_a, elements_a)]
    layout_a = calculate_layout(regions_a)

    elements_b, _, _ = find_table_elements(img_b, partitioning=True)
    clusters_b, _ = clustering(elements_b, n_jobs=N_CORES, alpha=alpha, beta=beta, gamma=gamma, radius=radius)
    regions_b = [Region(f_b, img=img_b, **r) for r in find_cluster_edges(clusters_b, elements_b)]
    layout_b = calculate_layout(regions_b)

    sim = parallel_layout_similarity(layout_a, layout_b)

    print("Regions detected in file", f_a)
    for i,r in enumerate(reversed(regions_a)):
        print("Region",i,r.top_lx, "-", r.bot_rx)

    print("\nRegions detected in file", f_b)
    for i,r in enumerate(reversed(regions_b)):
        print("Region",i,r.top_lx, "-", r.bot_rx)

    print("The similarity of their layout is", sim)

    if printimages:
        plt.ion()
        s_a = Spreadsheet(f_a)
        s_a.clusters = regions_a
        s_a.layout = layout_a
        s_a.print_image(img_a,save_path = "./")
        s_a.print_layout(save_path="./")

        s_b = Spreadsheet(f_b)
        s_b.clusters = regions_b
        s_b.print_image(img_b,save_path = "./")
        s_b.layout = layout_b
        s_b.print_layout(save_path="./")

    input("Press any key to exit")

if __name__ == "__main__":
    main()
