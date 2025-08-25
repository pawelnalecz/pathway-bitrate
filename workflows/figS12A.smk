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
include: "generic/_neighbors.smk"


# RULES

        
rule fig_S12A:
    input:
        **{
            f"neighbors_responding_fractions_{well_id}": 
                f'cache/neighbors/per_well/{well_id}/neighbors_responding_fractions_{dataset_id}_{model_id}_{train_id}_{test_id}.csv'
            for set_id, _ in set_types_and_colors
            for well_id in SET_ID_TO_WELL_IDS[set_id]
        },
    output:
        svg='figures/panels/figS12A.svg',
        png='figures/panels/figS12A.png',
    run:

        fig, axs = subplots_from_axsize(ncols=len(set_types_and_colors), axsize=(2,2), sharex=True, sharey=True, bottom=.6, top=.6, left=1., right=1.)

        for ax, (set_id, color) in zip(axs, set_types_and_colors):
            neighbors_responding_fractions = pd.concat([
                    pd.read_csv(input[f"neighbors_responding_fractions_{well_id}"], index_col=0)
                    for well_id in SET_ID_TO_WELL_IDS[set_id]
                ],
                names=['well_id'],
                keys=SET_ID_TO_WELL_IDS[set_id],
            )
            neighbors_responding_fractions.index.rename(['well_id', 'is_responding'], inplace=True)
            neighbors_responding_fractions.stack().unstack('is_responding').unstack('well_id')[True].plot(color=color, alpha=.3, lw=1, ax=ax)
            neighbors_responding_fractions.stack().unstack('is_responding').unstack('well_id')[False].plot(color=color, ls=':', alpha=.3, lw=1, ax=ax)
            neighbors_responding_fractions.groupby('is_responding').mean().loc[True].plot(color=color, lw=2, ax=ax)
            neighbors_responding_fractions.groupby('is_responding').mean().loc[False].plot(color=color, ls=':', lw=2, ax=ax)
            # neighbors_responding_fractions.groupby(level=0).plot(color=color, lw=2, ax=ax)
            handles, _ = ax.get_legend_handles_labels()
            ax.legend(labels=['around transmitting cell', 'around non-transmitting cell'], handles=handles[-2:])
            # ax.get_legend().set_visible(False)
            ax.set_title(set_to_label[set_id], fontsize='medium') #.replace('\n', ' + ')
        
                    
        fig.supylabel('fraction of responding cells')
        fig.supxlabel('neighbor No')

        fig.savefig(str(output.svg))
        fig.savefig(str(output.png), dpi=300)

