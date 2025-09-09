import numpy as np
import pandas as pd
from subplots_from_axsize import subplots_from_axsize

import matplotlib

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
    # ('main+STE1+0uM',        'slategray'),
    ('main+STE1+criz03uM',   'skyblue'),
    ('main+STE1+criz1uM',   'deepskyblue'),
    ('main+STE1+criz3uM',   'dodgerblue'),
    ('main+BEAS2B+0uM',      'goldenrod'),
    ('main+BEAS2B+cycl1uM',     'gold'),
    ('main+BEAS2B+tram05uM',     'red'),
    # ('main+BEAS2B+criz03uM', 'sandybrown'),
    # ('main+BEAS2B+tram05uMcycl1uM', 'navajowhite'),
]


# INCLUDE CORE RULES

include: "generic/_core_rules.smk"
include: "generic/_neighbors.smk"


# RULES

        
rule fig_31:
    input:
        **{
            f"neighbors_responding_fractions_{well_id}": 
                f'cache/neighbors/per_well/{well_id}/neighbors_responding_fractions_{dataset_id}_{model_id}_{train_id}_{test_id}.csv'
            for set_id, _ in set_types_and_colors
            for well_id in SET_ID_TO_WELL_IDS[set_id]
        },
    output:
        svg='figures/panels/fig31.svg',
        png='figures/panels/fig31.png',
    run:

        nrows = 2

        fig, axs = subplots_from_axsize(
            nrows=nrows,
            ncols=(len(set_types_and_colors) - 1) // nrows + 1,
            axsize=(1.4,1.4),
            bottom=.4,
            top=.4,
            left=.8,
            right=.8,
            wspace=.1,
            squeeze=False,
        )

        for ax, (set_id, color) in zip(axs.flatten(), set_types_and_colors):
            neighbors_responding_fractions = pd.concat([
                    pd.read_csv(input[f"neighbors_responding_fractions_{well_id}"], index_col=0)
                    for well_id in SET_ID_TO_WELL_IDS[set_id]
                ],
                names=['well_id'],
                keys=SET_ID_TO_WELL_IDS[set_id],
            )

            neighbors_responding_fractions.index.rename(['well_id', 'is_responding'], inplace=True)
            neighbors_responding_fractions_grouped = neighbors_responding_fractions.groupby('is_responding').mean()
            neighbors_responding_fractions_grouped.columns = neighbors_responding_fractions_grouped.columns.astype(int)
            neighbors_responding_fractions_grouped.loc[True].plot(color=color, lw=2, ax=ax)
            neighbors_responding_fractions_grouped.loc[False].plot(color=color, ls=':', lw=2, ax=ax)
            # handles, _ = ax.get_legend_handles_labels()
            # ax.legend(labels=['around transmitting cell', 'around non-transmitting cell'], handles=handles[-2:])
            # ax.get_legend().set_visible(False)
            ax.set_title(set_to_label[set_id].replace('\n', ' + '), fontsize='medium') #.replace('\n', ' + ')
            ax.set_ylim(0,1)
            ax.yaxis.set_major_formatter("{:.0%}".format)
            ax.set_xticks([1,5,10,15,20,25])

        handles, _ = ax.get_legend_handles_labels()
        ax.legend(labels=['around \ntransmitting', 'around \nnon-transmitting'], handles=handles[-2:], loc='lower right')
        
        for ax in axs[:, 1:].flatten():
            ax.set_yticklabels([])
            ax.set_ylabel('')

        for ax in axs[:-1].flatten():
            # ax.set_xticklabels([])
            ax.set_xlabel('')
                    
        fig.supylabel('fraction of transmitting cells', fontsize='medium')
        fig.supxlabel('$k$th nearest neighbor', fontsize='medium')

        fig.savefig(str(output.svg))
        fig.savefig(str(output.png), dpi=300)

