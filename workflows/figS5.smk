import matplotlib
from subplots_from_axsize import subplots_from_axsize
import pandas as pd

from generic._defaults import (
    DATA_MANAGER,
    WELLS_SELECTED,
    SET_TYPES,
    MODEL_IDS,
    DATASET_IDS,
)

from src import fig_style
from src.fig_layout import well_info_to_color, well_info_to_label, row_to_pos
from matplotlib.ticker import MultipleLocator
from config.local_config import OUTPUT_PATH

matplotlib.use("agg")
workdir: OUTPUT_PATH


# CONFIGS

TRAIN_IDS = ['main-q0', 'main-self-q0']
TEST_IDS = ['q1']

# INCLUDE CORE RULES

include: "generic/_core_rules.smk"
include: "generic/_join_results.smk"

# RULES

rule fig_S5:
    input:
        mi_all='figures/data/figS5/mi_all.csv'
    output:
        svg='figures/panels/figS5.svg',
        png='figures/panels/figS5.png',
    resources:
        mem_gib=1
    run:
        mi_all = pd.read_csv(input.mi_all)

        fig, ax = subplots_from_axsize(axsize=(2, 2), right=2.8)

        pivot = mi_all.groupby(['well_id', 'train_id_0'])['mi_ce'].mean().unstack('train_id_0')
        colors = pd.Series({well_id: well_info_to_color(well_info) for well_id, well_info in WELLS_SELECTED.loc[pivot.index].iterrows()})
        labels = pd.Series({well_id: well_info_to_label(well_info) for well_id, well_info in WELLS_SELECTED.loc[pivot.index].iterrows()})
        poss = pd.Series({well_id: row_to_pos(well_info) for well_id, well_info in WELLS_SELECTED.loc[pivot.index].iterrows()})
        pivot.plot.scatter('main-q0', 'main-self-q0', ax=ax, ec='k', alpha=.6, c=colors)
        ax.plot([0, 1.1*pivot.max().max()], [0, 1.1*pivot.max().max()], ls=':', color='k', alpha=.2)

        ax.set_xlabel('bitrate [bit/h]\ntrained on all experiments')
        ax.set_ylabel('bitrate [bit/h]\ntrained on leave-one-out')

        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_major_locator(MultipleLocator(5))

        for _, (pos, label, color) in poss.drop_duplicates().sort_values().to_frame('pos').join(labels.to_frame('labels')).join(colors.to_frame('colors')).iterrows():
            ax.plot([], marker='o', ms=4, markeredgecolor='k', ls='none', alpha=.6, color=color, label=label)
        ax.legend(loc=(1.1, 0))


        fig.savefig(str(output.svg))
        fig.savefig(str(output.png), dpi=300)


