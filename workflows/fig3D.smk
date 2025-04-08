import numpy as np
import pandas as pd

import matplotlib
from matplotlib import ticker
from subplots_from_axsize import subplots_from_axsize

from generic._defaults import (
    DATA_MANAGER,
    WELLS_SELECTED,
    SET_TYPES
)

from src.fig_layout import set_to_label
from src import fig_style
from config.local_config import OUTPUT_PATH

matplotlib.use("agg")
workdir: OUTPUT_PATH

# CONFIGS

dataset_id = 'ls+cell+inhs'
model_id = 'nn'
train_id = 'main-q0'
test_id = 'q1'

DATASET_IDS = [dataset_id]
MODEL_IDS = [model_id]
TRAIN_IDS = [train_id]
TEST_IDS = [test_id]

set_types_and_colors = [
    ('main+STE1+0uM',        'slategray'),
    ('main+STE1+criz03uM',   'deepskyblue'),
    ('main+BEAS2B+0uM',      'goldenrod'),
    ('main+BEAS2B+cycl1uM',     'gold'),
    ('main+BEAS2B+tram05uM',     'red'),
    # ('main+BEAS2B+criz03uM', 'sandybrown'),
    # ('main+BEAS2B+tram05uMcycl1uM', 'navajowhite'),
]


# INCLUDE CORE RULES

include: "generic/_core_rules.smk"
include: "generic/_combine_plots.smk"


# RULES

rule fig_3D:
    input:
        **{
            f"{set_id}_tracks_info": f'cache/preprocessed/per_set/{set_id}/tracks_info.csv.gz'
            for set_id, _ in set_types_and_colors
        },
        **{
            f"{set_id}_tracks_mi": f'cache/tracks_mi/per_set/{set_id}/tracks_mi_{dataset_id}_{model_id}_{train_id}_{test_id}.csv.gz'
            for set_id, _ in set_types_and_colors
        },
    output:
        svg='figures/panels/fig3D.svg',
        png='figures/panels/fig3D.png',
    run:

        fig, axs = subplots_from_axsize(
            # axsize=(.9, .9),
            axsize=(1.05, 1.05),
            wspace=[.1, .4, .1, .1],
            hspace=.7,
            ncols=len(set_types_and_colors),
            bottom=.35,
            nrows=1,
            top=.5,
            left=.5,
        )

        for ax, (set_id, color) in zip(axs.flatten(), set_types_and_colors):
            well_ids = SET_ID_TO_WELL_IDS[set_id] 
            experiments = [DATA_MANAGER.get_experiment(well_id) for well_id in well_ids]
            seconds_per_timepoint = DATA_MANAGER.get_seconds_per_timepoint_for_experiment_list(experiments)
            bph = 60 * 60 / seconds_per_timepoint / np.log(2)
            index_col = ['well_id', 'track_id']
     
            tracks_mi = pd.read_csv(input[f'{set_id}_tracks_mi'], index_col=index_col)
            tracks_info = pd.read_csv(input[f'{set_id}_tracks_info'], index_col=index_col)
            tracks_mi_and_info = tracks_mi.join(tracks_info)

            col = 'average_response_amplitude_over_reference'
            xlabel = 'average response amplitude [log fold change]'

            ax.scatter(
                tracks_mi_and_info[col],
                tracks_mi_and_info['mi_cross_per_slot'] * bph,
                s=4, #tracks_mi_and_info['slots'] / 100,
                color='brown',
                edgecolor='none',
                alpha=tracks_mi_and_info['slots'] / 10000,
            )

            ax.xaxis.set_major_locator(ticker.MultipleLocator(.5))

            ax.set_xlim(-.4, .8)
            ax.set_ylim(-5, 18)

            ax.set_title(set_to_label[set_id], fontsize='medium') #.replace('\n', ' + ')

        
        for ax in axs.flatten()[1:]:
            ax.set_yticks([])
        
        fig.supylabel('bitrate [bit/h]', fontsize='medium')
        fig.supxlabel(xlabel, fontsize='medium')


        fig.savefig(str(output.svg))
        fig.savefig(str(output.png), dpi=300)

