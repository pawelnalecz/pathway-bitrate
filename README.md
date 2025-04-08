# pathway-bitrate
Python scripts to compute bitrate achieved by a biological signaling pathway. Used in Nalecz-Jawecki et al. 2025

# Prerequisites 
* Python >= 3.10
* ShuttleTracker (only for image processing) [download from https://pmbm.ippt.pan.pl/web/ShuttleTracker]

# Installation
* Open your terminal (e.g., bash, zsh, cmd, PowerShell). (On Windows, press Windows+s and search for cmd)
* Go to the main folder of the package
```bash
cd /some_path/pathway-bitrate
```
* (Optional but recommended) Create a Python virtual environment [see https://docs.python.org/3/library/venv.html] and activate it
* Install requirements
```bash
pip install -r requirements.txt
```

# Repoducing results from the paper
* Download the data from Zenodo [link].
* Open local_config.ini and enter the path to the downlowaded data as well as the output directory
* Run the workflow for generating a particular panel, e.g.,
```bash
snakemake --snakefile workflows/fig2A.smk --cores all
```
* The will appear in the figures/panels subdirectory of your output folder.