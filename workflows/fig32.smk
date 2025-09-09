import numpy as np
import pandas as pd

import matplotlib
from matplotlib.ticker import MultipleLocator
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

base_dataset_id = 'ls+cell+inhs'
model_id = 'nn'
train_id = 'main-q0'
test_id = 'q1'


DATASET_SUFFIXES = ['', '--early', '--late', '--long']
DATASET_IDS =  [base_dataset_id + suffix for suffix in DATASET_SUFFIXES]# ['ls+cell+inhs', 'ls+cell+inhs--early', 'ls+cell+inhs--late', 'ls+cell+inhs--long']
MODEL_IDS = [model_id]
TRAIN_IDS = [train_id]
TEST_IDS = [test_id]

set_types_and_colors = [
    # ('main+STE1+0uM',        'slategray'),
    ('main+STE1+criz03uM',   'skyblue'),
    # ('main+STE1+criz1uM',   'deepskyblue'),
    # ('main+STE1+criz3uM',   'dodgerblue'),
    ('main+BEAS2B+0uM',      'goldenrod'),
    # ('main+BEAS2B+cycl1uM',     'gold'),
    # ('main+BEAS2B+tram05uM',     'red'),
    # # ('main+BEAS2B+criz03uM', 'sandybrown'),
    # # ('main+BEAS2B+tram05uMcycl1uM', 'navajowhite'),
]


# INCLUDE CORE RULES

include: "generic/_core_rules.smk"
include: "generic/_combine_plots.smk"


# RULES

rule fig_32:
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
        svg='figures/panels/fig32.svg',
        png='figures/panels/fig32.png',
    run:

        nrows = 1
        fig, axs = subplots_from_axsize(
            axsize=(2.05, 2.05),
            wspace=.7,
            hspace=.7,
            nrows=nrows,
            ncols=(len(set_types_and_colors) - 1) // nrows + 1,
            bottom=.5,
            top=.5,
            left=.6,
            right=.6,
            sharex=True,
            sharey=True,
        )

        xsuffix = '--early'
        ysuffix = '--late'

        colx = 'mi_cross_per_slot' + xsuffix
        coly = 'mi_cross_per_slot' + ysuffix

        xlabel = 'bitrate – detection by dip [bit/h]'
        ylabel = 'bitrate – detection by peak [bit/h]'

        colors = pd.Series(['red', 'green'], index=[False, True])


        for ax, (set_id, color) in zip(axs.flatten(), set_types_and_colors):
            well_ids = SET_ID_TO_WELL_IDS[set_id] 
            experiments = [DATA_MANAGER.get_experiment(well_id) for well_id in well_ids]
            seconds_per_timepoint = DATA_MANAGER.get_seconds_per_timepoint_for_experiment_list(experiments)
            bph = 60 * 60 / seconds_per_timepoint / np.log(2)
            index_col = ['well_id', 'track_id']

              
            tracks_mi = pd.concat(
                (
                    pd.read_csv(input[f'{set_id}_{dataset_id}_tracks_mi'], index_col=index_col)
                        .add_suffix(dataset_suffix)
                    for dataset_suffix, dataset_id in zip(DATASET_SUFFIXES, DATASET_IDS)
                ),
                axis='columns',
            )
            tracks_info = pd.read_csv(input[f'{set_id}_tracks_info'], index_col=index_col)
            tracks_mi_and_info = tracks_mi.join(tracks_info)


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
          
            transmitting_thresholds = pd.Series(
                [
                    tracks_mi_and_info[tracks_mi_and_info['is_transmitting' + dataset_suffix]]['mi_cross_per_slot' + dataset_suffix].min() 
                    for dataset_suffix in DATASET_SUFFIXES
                ], 
                index=DATASET_SUFFIXES,
            ) * bph
            
            ax.axvline(transmitting_thresholds[xsuffix], ls='--', color='k', alpha=.3)
            ax.axhline(transmitting_thresholds[ysuffix], ls='--', color='k', alpha=.3)

            xmin, xmax = -5, 17
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(xmin, xmax)

            for valx, valy, posx, posy in [
                (False, True , (xmin + transmitting_thresholds[xsuffix]) / 2, (xmax + transmitting_thresholds[ysuffix]) / 2),
                (True , True , (xmax + transmitting_thresholds[xsuffix]) / 2, (xmax + transmitting_thresholds[ysuffix]) / 2),
                (False, False, (xmin + transmitting_thresholds[xsuffix]) / 2, (xmin + transmitting_thresholds[ysuffix]) / 2),
                (True , False, (xmax + transmitting_thresholds[xsuffix]) / 2, (xmin + transmitting_thresholds[ysuffix]) / 2),
            ]:
                ax.annotate(
                    f"{fractions_transmitting.loc[valx, valy]: .1%}",
                    (posx, posy), 
                    # xycoords=('axes fraction'),
                    fontweight='bold',
                    fontsize='large',
                    horizontalalignment='center',
                    verticalalignment='center',
                )


            ax.xaxis.set_major_locator(MultipleLocator(5))
            ax.yaxis.set_major_locator(MultipleLocator(5))

            ax.set_title(set_to_label[set_id].replace('\n', ' + '), fontsize='medium') #.replace('\n', ' + ')

        
        fig.supylabel(ylabel, fontsize='medium')
        fig.supxlabel(xlabel, fontsize='medium')

        fig.savefig(str(output.svg))
        fig.savefig(str(output.png), dpi=300)

