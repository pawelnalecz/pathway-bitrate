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
from src.fig_layout import well_info_to_color
from config.local_config import OUTPUT_PATH

matplotlib.use("agg")
workdir: OUTPUT_PATH


# CONFIGS

WELLS_SELECTED = DATA_MANAGER.wells[
    DATA_MANAGER.wells['experiment'].isin([
        '2024-07-30-STE1',
    ])
]


# INCLUDE CORE RULES

include: "generic/_core_rules.smk"
include: "generic/_per_to_nuc_translocation.smk"

# RULES

rule fig_S3:
    input:
        expand(
            'cache/per_to_nuc/per_well/{well_id}/mean_per_to_nuc_trajectory.csv.gz',
            well_id=WELLS_SELECTED.index,
        ),
    output:
        svg='figures/panels/figS3.svg',
        png='figures/panels/figS3.png',
    resources:
        mem_gib=lambda wc, input: 2 * len(input),
    run:
        well_infos = WELLS_SELECTED.join(DATA_MANAGER.experiments, on='experiment', rsuffix='_')

        mean_trajectories = pd.concat(
            [pd.read_csv(path, index_col='time_in_seconds')['log_per_to_nuc_translocation']
                for path in input
            ],
            axis='columns',
            names=['well_id'],
            keys=WELLS_SELECTED.index,
        )

        xmin = 0
        xmax = 400 * 60

        fig, ax = subplots_from_axsize(axsize=(6,2.2), left=.7)
        conditions = []
        for well_id in mean_trajectories:
            label = f"{WELLS_SELECTED.loc[well_id]['inh_crizotinib']}uM" 
            if label in conditions:
                label = '_'
            else:
                conditions.append(label)
            
            mean_trajectories[well_id].plot(
                ax=ax,
                color=well_info_to_color(WELLS_SELECTED.loc[well_id]), 
                alpha=.6, 
                label=label,
                )
            
        ax.xaxis.set_major_locator(MultipleLocator(1800))
        ax.xaxis.set_minor_locator(MultipleLocator(600))
        ax.xaxis.set_major_formatter(lambda x, pos: f"{x // 3600:.0f}" if not x % 3600 else '')
        ax.set_xlim(xmin, xmax)
        

        ax.yaxis.set_major_locator(MultipleLocator(.1))

        ax.set_ylabel('log ERK-KTR translocation (cyt:nuc)')
        ax.set_xlabel('time [h]')
        ax.legend(title='ALKi', loc='lower right')
        fig.savefig(str(output.svg))
        fig.savefig(str(output.png), dpi=300)

