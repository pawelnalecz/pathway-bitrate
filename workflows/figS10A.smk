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
train_id = 'maincellinh-q0'
test_id = 'q1'


DATASET_SUFFIXES = ['', '--early', '--late', '--long']
DATASET_IDS =  [dataset_id + suffix for suffix in DATASET_SUFFIXES]# ['ls+cell+inhs', 'ls+cell+inhs--early', 'ls+cell+inhs--late', 'ls+cell+inhs--long']
MODEL_IDS = [model_id]
TRAIN_IDS = [train_id]
TEST_IDS = [test_id]

set_types_and_colors = [
    ('main+STE1+0uM',        'slategray'),
    ('main+STE1+criz03uM',   'deepskyblue'),
    ('main+BEAS2B+0uM',      'goldenrod'),
    ('main+BEAS2B+cycl1uM',     'gold'),
    ('main+BEAS2B+tram05uM',     'red'),
    # # ('main+BEAS2B+criz03uM', 'sandybrown'),
    # # ('main+BEAS2B+tram05uMcycl1uM', 'navajowhite'),
]


# INCLUDE CORE RULES

include: "generic/_core_rules.smk"
include: "generic/_combine_plots.smk"


# RULES

rule fig_S10A:
    input:
        **{
            f"{set_id}_tracks_info": f'cache/preprocessed/per_set/{set_id}/tracks_info.csv.gz'
            for set_id, _ in set_types_and_colors
        },
        **{
            f"{set_id}_{dataset_id}_tracks_mi": f'cache/tracks_mi/per_set/{set_id}/tracks_mi_{dataset_id}_{model_id}_{train_id}_{test_id}.csv.gz'
            for set_id, _ in set_types_and_colors for dataset_id in DATASET_IDS
        },
    output:
        svg='figures/panels/figS10A.svg',
        png='figures/panels/figS10A.png',
    run:

        fig, axs = subplots_from_axsize(
            # axsize=(.9, .9),
            axsize=(3.05, 3.05),
            # wspace=[.1, .4, .1, .1],
            hspace=.7,
            ncols=len(set_types_and_colors),
            bottom=.5,
            nrows=1,
            top=.5,
            left=1.,
            sharex=True,
            sharey=True,
        )

        for ax, (set_id, color) in zip(axs.flatten(), set_types_and_colors):
            well_ids = SET_ID_TO_WELL_IDS[set_id] 
            experiments = [DATA_MANAGER.get_experiment(well_id) for well_id in well_ids]
            seconds_per_timepoint = DATA_MANAGER.get_seconds_per_timepoint_for_experiment_list(experiments)
            bph = 60 * 60 / seconds_per_timepoint / np.log(2)
            index_col = ['well_id', 'track_id']

              
            tracks_mi = pd.concat(
                    (pd.read_csv(input[f'{set_id}_{dataset_id}_tracks_mi'], index_col=index_col).add_suffix(dataset_suffix) for dataset_suffix, dataset_id in zip(DATASET_SUFFIXES, DATASET_IDS)),
                    axis='columns',
                )
            tracks_info = pd.read_csv(input[f'{set_id}_tracks_info'], index_col=index_col)
            tracks_mi_and_info = tracks_mi.join(tracks_info)

            xsuffix = '--early'
            ysuffix = '--late'

            colx = 'mi_cross_per_slot' + xsuffix
            coly = 'mi_cross_per_slot' + ysuffix
            xlabel = 'bitrate 1-2 min (dip) [bit/h]'
            ylabel = 'bitrate 7-12 min (peak) [bit/h]'

            colors = pd.Series(['red', 'green'], index=[False, True])

            ax.scatter(
                tracks_mi_and_info[colx] * bph,
                tracks_mi_and_info[coly] * bph,
                s=4, #tracks_mi_and_info['slots'] / 100,
                c=colors.reindex(tracks_mi_and_info['is_transmitting']),
                edgecolor='none',
                alpha=tracks_mi_and_info['slots'] / 10000,
            )

            corr = tracks_mi_and_info.corr()

            fractions_transmitting = tracks_mi_and_info.groupby(['is_transmitting' + xsuffix, 'is_transmitting' + ysuffix])['slots'].sum() / tracks_mi_and_info['slots'].sum()
            ax.annotate(
                f"{fractions_transmitting.loc[False, True ]: 10.1%}" 
                f"{fractions_transmitting.loc[True , True ]: 10.1%}" "\n"
                f"{fractions_transmitting.loc[False, False]: 10.1%}" 
                f"{fractions_transmitting.loc[True , False]: 10.1%}" "\n\n"
                f"corr (bitrates) = {corr['mi_cross_per_slot' + xsuffix]['mi_cross_per_slot' + ysuffix]:.2f}" "\n"
                f"corr (transmitting) = {corr['is_transmitting' + xsuffix]['is_transmitting' + ysuffix]:.2f}" "\n"
                , 
             (0.55, 0.6), xycoords=('axes fraction'),
             fontweight='bold',)
            
            transmitting_thresholds = pd.Series(
                [
                    tracks_mi_and_info[tracks_mi_and_info['is_transmitting' + dataset_suffix]]['mi_cross_per_slot' + dataset_suffix].min() 
                    for dataset_suffix in DATASET_SUFFIXES
                ], 
                index=DATASET_SUFFIXES,
                ) * bph
            
            ax.axvline(transmitting_thresholds[xsuffix], ls='--', color='k', alpha=.3)
            ax.axhline(transmitting_thresholds[ysuffix], ls='--', color='k', alpha=.3)

            # ax.xaxis.set_major_locator(ticker.MultipleLocator(.5))

            ax.set_xlim(-5, 17)
            ax.set_ylim(-5, 17)

            ax.set_title(set_to_label[set_id], fontsize='medium') #.replace('\n', ' + ')

            # ax.set_xlabel(xlabel)
            # ax.set_ylabel(ylabel)

        
        # for ax in axs.flatten()[1:]:
        #     ax.set_yticks([])
        
        fig.supylabel(ylabel, fontsize='large')
        fig.supxlabel(xlabel, fontsize='large')


        fig.savefig(str(output.svg))
        fig.savefig(str(output.png), dpi=300)

