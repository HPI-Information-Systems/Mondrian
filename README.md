# mondrian_sigmod
Mondrian repository for SIGMOD 2021 submission

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

## Experiment scripts

The region_detection.py scripts can be used to run the experiments on the multiregion detection components.
The template_recognition.py scripts can be used to run the experiments on the multiregion detection components.
They both require the input datasets of csv files to be stored in a folder named "./res/{dataset-name}/csv" and their annotations in the folder "./res/{dataset-name}/annotations".
The region_detection.py script requires the annotated regions to be in a file named "annotations_elements.json" inside the annotation folder, while the template_recognition.py script requires additionally a file named "annotations_templates.json".

The output of the scripts will be produced in a "result" folder where the script are launched.
To correcly launch the template_recognition.py script, there need to be results produced from the multiregion detection script.

