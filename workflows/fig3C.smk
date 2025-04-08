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

rule fig_3C:
    input:
        transmitting_analysis_all='figures/data/fig3BC/transmitting_analysis_all.csv'
    output:
        svg='figures/panels/fig3C.svg',
        png='figures/panels/fig3C.png',

    resources:
        mem_gib=1
    run:
        mi_all = pd.read_csv(input.transmitting_analysis_all)
        
        fig, ax = subplots_from_axsize(axsize=(1.4, 6. * 7.5/9.5), top=0.4, right=.6, left=.1, bottom=.8)
        
        pe = mi_all['pe_transmitting'].max()

        plot_mi(ax,
            mi_all,
            field='mi_ce_transmitting',
            title='average bitrate\nin transmitting cells [bit/h]',
            plot_labels=False,
            means_format_str="{mean:.1f} $\pm$ {sem:.1f} bit/h",
            ymin=-0.5,
            ymax=1.1*pe,
            marker='o',
            )
        # ax.set_title(letter, horizontalalignment='left')

        ax.axvline(0, ls='-', alpha=.2, color='grey')
        ax.axvline(pe, ls=':', alpha=.4, color='grey')
        # ax.axvline(pe, ls='-', alpha=.4, lw=.5,  color='grey')

        fig.savefig(str(output.svg))
        fig.savefig(str(output.png), dpi=300)

