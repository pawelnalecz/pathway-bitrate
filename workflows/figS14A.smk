import numpy as np
import pandas as pd

import matplotlib
from matplotlib import ticker
from subplots_from_axsize import subplots_from_axsize

from generic._defaults import (
    DATA_MANAGER,
    WELLS_SELECTED,
    SET_TYPES,
    DATASET_IDS,
    MODEL_IDS,
    TRAIN_IDS,
    TEST_IDS,
)

from src.fig_layout import set_to_label
from src import fig_style
from config.local_config import OUTPUT_PATH

matplotlib.use("agg")
workdir: OUTPUT_PATH

# CONFIGS

dataset_id = DATASET_IDS[0]
model_id = MODEL_IDS[0]
train_id = TRAIN_IDS[0]
test_id = TEST_IDS[0]

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
include: "generic/_neighbors.smk"


# RULES

rule fig_S14A:
    input:
        **{
            f"{set_id}_tracks_mi": f'cache/tracks_mi/per_set/{set_id}/tracks_mi_{dataset_id}_{model_id}_{train_id}_{test_id}.csv.gz'
            for set_id, _ in set_types_and_colors
        },
        **{
            f"{set_id}_track_confluence": f'cache/neighbors/per_set/{set_id}/track_confluence.csv.gz'
            for set_id, _ in set_types_and_colors
        },
    output:
        svg='figures/panels/figS14A.svg',
        png='figures/panels/figS14A.png',
    run:

        fig, axs = subplots_from_axsize(
            # axsize=(.9, .9),
            axsize=(2.05, 2.05),
            # wspace=[.1, .4, .1, .1],
            hspace=.7,
            ncols=len(set_types_and_colors),
            bottom=1.,
            nrows=1,
            top=1.,
            left=1.,
            right=.1,
            # sharex=True,
            sharey=True,
        )

        for ax, (set_id, color) in zip(axs.flatten(), set_types_and_colors):
            well_ids = SET_ID_TO_WELL_IDS[set_id] 
            experiments = [DATA_MANAGER.get_experiment(well_id) for well_id in well_ids]
            seconds_per_timepoint = DATA_MANAGER.get_seconds_per_timepoint_for_experiment_list(experiments)
            bph = 60 * 60 / seconds_per_timepoint / np.log(2)
            index_col = ['well_id', 'track_id']
     
            tracks_mi = pd.read_csv(input[f'{set_id}_tracks_mi'], index_col=index_col)
            track_confluence = pd.read_csv(input[f'{set_id}_track_confluence'], index_col=index_col)
            tracks_mi_with_confluence = tracks_mi.join(track_confluence)

            col = 'average_response_amplitude_over_reference'
            xlabel = 'average response amplitude [log fold change]'

            ax.scatter(
                tracks_mi_with_confluence['confluence'],
                tracks_mi_with_confluence['mi_cross_per_slot'] * bph,
                s=4, #tracks_mi_and_info['slots'] / 100,
                color='brown',
                edgecolor='none',
                alpha=tracks_mi_with_confluence['slots'] / 10000,
            )

            tracks_mi_with_confluence_per_well = tracks_mi_with_confluence.groupby('well_id')[['confluence', 'mi_cross_per_slot']].mean()

            ax.scatter(
                tracks_mi_with_confluence_per_well['confluence'],
                tracks_mi_with_confluence_per_well['mi_cross_per_slot'] * bph,
                s=10, #tracks_mi_and_info['slots'] / 100,
                color='red',
                edgecolor='none',
            )

            ax.annotate(f"per_welll corr: {tracks_mi_with_confluence_per_well.corr().loc['confluence', 'mi_cross_per_slot']}:.2f    ", (0.3, 0.1), xycoords='axes fraction')

            corr = tracks_mi_with_confluence[['confluence', 'mi_cross_per_slot']].corr().loc['confluence', 'mi_cross_per_slot']
            ax.annotate(f"corr = {corr:.2f}", (0.3,.6), xycoords='axes fraction')

            ax.set_title(set_to_label[set_id], fontsize='medium') #.replace('\n', ' + ')

        
        
        fig.supylabel('bitrate [bit/h]', fontsize='medium')
        fig.supxlabel('confluence [cells/px^2]', fontsize='medium')


        fig.savefig(str(output.svg))
        fig.savefig(str(output.png), dpi=300)

