import matplotlib
from subplots_from_axsize import subplots_from_axsize
import pandas as pd

from generic._defaults import (
    DATA_MANAGER,
    WELLS_SELECTED,
    SET_TYPES,
    DATASET_IDS,
    MODEL_IDS,
    TRAIN_IDS,
    TEST_IDS,
)

from src.jax_plots import plot_mi
from src import fig_style
from config.local_config import OUTPUT_PATH

matplotlib.use("agg")
workdir: OUTPUT_PATH

# CONFIGS

WELLS_SELECTED = WELLS_SELECTED[
    (
        (WELLS_SELECTED['cell_line'] == 'BEAS2B') 
      & (WELLS_SELECTED['inh_crizotinib'] == 0) 
      & ((WELLS_SELECTED['inh_trametinib'] == 0) | (WELLS_SELECTED['inh_cyclosporin'] == 0))
    ) |
    (
        (WELLS_SELECTED['cell_line'] == 'STE1') 
    )
]


# INCLUDE CORE RULES

include: "generic/_core_rules.smk"
include: "generic/_join_results.smk"


# RULES

rule fig_3B:
    input:
        transmitting_analysis_all='figures/data/fig3BC/transmitting_analysis_all.csv'
    output:
        svg='figures/panels/fig3B.svg',
        png='figures/panels/fig3B.png',

    resources:
        mem_gib=1
    run:
        mi_all = pd.read_csv(input.transmitting_analysis_all)
        
        fig, ax = subplots_from_axsize(axsize=(1.4, 6.* 7.5/9.5), top=0.4, right=.6, left=.1, bottom=.8)

        plot_mi(ax,
            mi_all,
            field='fraction_transmitting',
            title='transmitting subpopulation size',
            plot_labels=False,
            means_format_str="{mean:.0%} $\pm$ {sem:.0%}",
            ymin=0.,
            ymax=1.,
            marker='o',
            )
        # ax.set_title(letter, horizontalalignment='left')

        ax.xaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")

        fig.savefig(str(output.svg))
        fig.savefig(str(output.png), dpi=300)

