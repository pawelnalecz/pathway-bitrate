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

from src.fig_layout import row_to_pos
from src import fig_style
from src import jax_plots
from config.local_config import OUTPUT_PATH

matplotlib.use("agg")
workdir: OUTPUT_PATH


# CONFIGS

TRAIN_IDS = ['main-q0']
TEST_IDS = ['q1']

# INCLUDE CORE RULES

include: "generic/_core_rules.smk"
include: "generic/_measures.smk"
include: "generic/_join_results.smk"

# RULES

rule fig_2B:
    input:
        measures_all='figures/data/fig2B/measures_all_q1.csv'
    output:
        svg='figures/panels/fig2B.svg',
        png='figures/panels/fig2B.png',
    resources:
        mem_gib=1
    run:
        measures_all = pd.read_csv(input.measures_all)
        measures_all['pos'] = measures_all.apply(row_to_pos, axis=1)

        fig, ax = subplots_from_axsize(axsize=(1.4, 6.), top=0.4, right=.6, left=.05, bottom=.8)

        jax_plots.plot_mi(
            ax, 
            measures_all, 
            field='average_response_amplitude_over_reference',
            title='average response amplitude\n[log fold change]',
            plot_labels=False,
            means_format_str="{mean:.2f} $\pm$ {sem:.2f}",
            marker='o',
            clip_on=False,
        )

        ax.axvline(0, ls='-', alpha=.2, color='grey')

        fig.savefig(str(output.svg))
        fig.savefig(str(output.png), dpi=300)


