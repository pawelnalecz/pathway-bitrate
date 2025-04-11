from matplotlib import pyplot as plt
import matplotlib
from matplotlib.ticker import MultipleLocator
from subplots_from_axsize import subplots_from_axsize
import pandas as pd
import numpy as np

from generic._defaults import (
    DATA_MANAGER,
    WELLS_SELECTED,
    SET_TYPES,
)

from src import fig_style
from config.local_config import OUTPUT_PATH

matplotlib.use("agg")
workdir: OUTPUT_PATH


# CONFIGS


TEST_SET_TYPES = ['main+cell+inh']

# INCLUDE CORE RULES

include: "generic/_core_rules.smk"
include: "generic/_per_to_nuc_translocation.smk"
include: "generic/_combine_plots.smk"

# RULES

rule all:
    input:
        expand(
            'plot_per_to_nuc_trajectories/combined/per_experiment/{experiment}/per_to_nuc_trajectory.png',
            experiment=WELLS_SELECTED['experiment'].unique(),
        ),
        expand(
            'plot_per_to_nuc_trajectories/combined/per_set_type/{set_type}/per_to_nuc_trajectory.png',
            set_type=TEST_SET_TYPES,
        )

rule per_to_nuc_trajectory:
    input:
        'cache/per_to_nuc/{per}/{well_or_set_id}/mean_per_to_nuc_trajectory.csv.gz',
    output:
        'plot_per_to_nuc_trajectories/single/{per}/{well_or_set_id}/per_to_nuc_trajectory.png',
    resources:
        mem_gib=lambda wc, input: 2 * len(input),
    run:
        well_infos = WELLS_SELECTED.join(DATA_MANAGER.experiments, on='experiment', rsuffix='_')

        mean_trajectories = pd.read_csv(str(input), index_col='time_in_seconds')['log_per_to_nuc_translocation']

        print(mean_trajectories)

        xmin = 0
        xmax = 400 * 60

        fig, ax = subplots_from_axsize(axsize=(6,2.2), left=.7)
        conditions = []
        mean_trajectories.plot(
            ax=ax,
            alpha=.6, 
            label=wildcards.well_or_set_id,
            )
        print(wildcards.well_or_set_id)
        print(mean_trajectories)
            
        ax.xaxis.set_major_locator(MultipleLocator(1800))
        ax.xaxis.set_minor_locator(MultipleLocator(600))
        ax.xaxis.set_major_formatter(lambda x, pos: f"{x // 3600:.0f}" if not x % 3600 else '')
        ax.set_xlim(xmin, xmax)
        

        ax.yaxis.set_major_locator(MultipleLocator(.1))
        ax.set_ylim(-1., 1.)
        ax.set_ylabel('log ERK-KTR translocation (cyt:nuc)')
        ax.set_xlabel('time [h]')
        ax.legend(loc='lower right')
        fig.savefig(str(output), dpi=300)

