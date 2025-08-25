# pathway-bitrate
Python scripts to compute bitrate achieved by a biological signaling pathway. Used in Nalecz-Jawecki et al. 2025

Created by Paweł Nałęcz-Jawecki and Frederic Grabowski at IPPT PAN, Warsaw, Poland.
Available under BSD 3-Clause License.

# Setup
### Prerequisites 
* Python >= 3.10
* ShuttleTracker (only for image processing) [download from https://pmbm.ippt.pan.pl/web/ShuttleTracker]

### Installation
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
* Open `config/local_config.py` with a text editor and enter the path to the downloaded data and the output directory.
* Run the Snakemake workflow to generate a particular panel, e.g.,
```bash
snakemake --snakefile workflows/fig2A.smk --cores all --resources mem-gib=32 --rerun-incomplete
```
* The results will appear in the figures/panels subdirectory of your output folder.

### Other workflows
The `workflow` directory contains other snakefiles (not starting with fig)

# Reusing the scripts for own research (advanced)
The following workflow was used in our paper:
* Protocol generation
* Biological experiments
* Image analysis and cell tracking
* Bitrate computation
The steps are largely independent from each other, and can be easlily substituted, as long as the interface files are properly structured.

## Protocol generation
* Use the script `protocol_generation/generate.py` to generate a distribution of pulse intervals. 
* Adjust it sensibly (e.g., to avoid an interval length to be sampled only once)
* Randomize the sequence of pulses using "protocol_generation/randomize.py"
NOTE: Although in principle different stimulation sequences may be used across experiments/replicates, 
some of the figure-generating scripts silently assume a consistent pulse sequence in all replicates
and may produce misleading results if the assumption is not satisfied.
* Create a separate folder for your data. Create a `_predefined_protocols` folder in it and put the `distibution.yaml` file in it. Rename the file to match the protocol name specified inside the YAML file

## Biological experiments
* Stimulate cells according to the generated protocol and image with a constant frequency.
* Detect nuclei and track them as described below.
* In your data folder, create a subfolder for each experiment and a `metadata.yaml` in it. Specify the required metadata and the pulse sequence, taking the metadata files from the paper as a guide.

## Image analysis and cell tracking
In our analysis, we used the in-lab application ShuttleTracker that you can find [here](https://pmbm.ippt.pan.pl/web/ShuttleTracker). 
However, any software can be used, as long as it outputs a csv/pkl file with single-cell quantifications. The file should contain `track_id` and `time_in_seconds` columns as index and a `nuc_<channel>_intensity_mean` field for each channel (receptor, reporter).
If you decide to use ShuttleTracker, you can use the following procedure:
* Save your imaging data in a separate folder for each well, with a single TIF file for each frame and channel. Keep the following naming convention `Img_t0000_ch0.tif`.
* Create a `shuttletracker_metadata.txt` file as described in Shuttletracker documentation.
* Open the folder with ShuttleTracker GUI and choose proper parameters for nuclei detection, perinuclei derivation and tracking. Save the parameters.
* Specify the path to ShuttleTracker executable file in `image_quantification/local_config.py` 
* Detect and track nuclei using the `track_cells` function from `image_quantifiction/track_cells.py`.
* Merge quantifications using the `merge_track` function from `image_quantification/merge_tracks.py`. You should obtain an `<replicate_name>-tracks.pkl.gz` file. Place it in your data folder and specify the path to it in the `metadata.yaml` file for each experiment.

## Bitrate estimation
* Open `config/local_config.py` and enter the path to the downloaded data and the output directory. Specify the type of input file used (CSV/PICKLE).
* Open `config/parameters.py` and specify the receptor and reporter channel names to match the channel names in the quantification file. You can also change the thresholds for each quality level.
* Bitrate is computed in four main steps: 
  * DATASET (dataset creation)
  * MODEL (neural network configuration)
  * TRAIN (network training, possibly in a few rounds that may include protocol optimization)
  * TEST (evaluation on test set and bitrate estimation)
  Parameters for each step are defined separately. In `config/configs.py`, you can predefine sets of parameter values for each step. You can choose which sets to use by including their name in the `<STEP_NAME>_CONFIGS` lists, either in `workflows/_defaults.py` or in the separate workflow files for each figure.
* Run the `mi_defaults.smk` to obtain bitrate estimation.
* For running other scripts, you may need to adjust figure layouts in `workflows/_combine_plots.smk`, `src/fig_layout.py` and in the concerned workflow. You may also want to edit the set types defined in `DataManager.set_type_to_well_ids()` (`src/data_manager.py`).
