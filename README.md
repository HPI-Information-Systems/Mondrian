# Detecting Multiregion Templates with Mondrian
Code repository associated with the paper "Detecting Layout Templates in Complex Multiregion Files" submitted to Volume 15 of PVLDB.

## Setup

The code was written using Python 3.8.
If using a local (or virtual) environment, install dependencies with
`pip install -r requirements.txt`
Alternatively, if using a conda distribution, use:
`conda env create --file mondrian.yml`

## Basic script
The basic.py script can be used to detect region boundaries in two csv files and compare their layout similarity.
From the command line, type:

`python3 basic.py`

The script will ask the user for the location of two comma delimited files.
Optional parameters that can be provided to this script are the hyperparameters for the multiregion detection phase (alpha, beta, gamma and the radius), as well as a custom delimiter for the .csv files (default is comma) and the option to print images of the detected regions/graph layouts.

For the full list of parameters, type:

`python3 basic.py --help`

## Multiregion detection experiment

The region_detection.py scripts can be used to run the experiments on the multiregion detection component of Mondrian.
It requires the input datasets of csv files to be stored in a folder named "./res/{dataset-name}/csv" and their annotations in the folder "./res/{dataset-name}/annotations".
The dataset name can be specified as a command line argument.
The alpha, beta, gamma hyperparameters can be configured with command line parameters.
An additional argument allows to execute the evaluation of the results and print the results.

The script allows different configurations to be tested, selecting them with command line arguments.
To run region detection using the connected component baseline and see the evaluation results, run:

`python3 region_detection.py --baseline --evaluate`

To test using a static radius R, run:

`python3 region_detection.py --static R --evaluate`

To test using a dynamic, optimal radius, run:

`python3 region_detection.py --dynamic --evaluate`

For the full list of parameters, type:

`python3 region_detection.py --help`

## Template inference experiment
The template_recognition.py script can be used to run the experiments on the multiregion inference component of Mondrian.

Like region_detection.py, this script requires the target regions to be in a file named "annotations_elements.json" inside the annotation folder as well as the gold standard templates file named "annotations_templates.json".
Additionally, it requires results produced from the multiregion detection script.
The input files are read from the "res/{dataset-name}/csv" folder and the output of the scripts are read/written to the "result" folder.
The dataset name can be specified as a command line argument.
The script allows to set the desired hyperparameters for the multiregion detection as well as the thresholds for region and template similarity.

Following the same names used in the region_detection.py script, the script must be run with an --experiment parameter to select the corresponding results computed by the multiregion detection stage.
To run template inference using the connected component baseline run:

`python3 template_search.py --experiment baseline`

To test using a static radius R, run:

`python3 template_search.py --experiment static --r R --evaluate`

To test using a dynamic, optimal radius, run:

`python3 template_search.py --experiment dynamic`

For the full list of parameters, type:

`python3 template_search.py --help`

## Datasets 
The folder "res" in the dataset contains two annotated dataset of spreadsheet files: DECO and FUSTE.
Their respective folders contain the files in .csv format and the annotations in .json format.

The file "annotation_elements.json" contains the file-level annotations for region boundaries in the form: 

    {
        "file1": {
            "n_regions": int,
            "regions": [
                {
                    "region_label": string,
                    "region_type": string,
                    "top_lx": [int,int], #the top-left coordinate of region boundary
                    "bot_rx": [int,int], #the bot-right coordingate of the region boundary
                  ...
                  }]},
          "file2": ...
    }

The file "annotations_template.json" contains dataset-level annotations of templates in the form: 

`{
  "template_name": ["file1", "file2", ...]
}`
