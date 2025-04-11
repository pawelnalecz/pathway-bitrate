import matplotlib
from subplots_from_axsize import subplots_from_axsize
import pandas as pd

from generic._defaults import (
    DATA_MANAGER,
    WELLS,
    SET_TYPES,
    DATASET_IDS,
    MODEL_IDS,
    TRAIN_IDS,
)

from src import fig_style
from src import jax_plots
from config.local_config import OUTPUT_PATH

matplotlib.use("agg")
workdir: OUTPUT_PATH


### PLOT RESPONSE AMPLITUDE VS STIMULATION STRENGTH

# CONFIGS

WELLS_SELECTED = WELLS[
    WELLS['experiment'].isin([
        '2023-08-28-BEAS2B--intensities',
        '2024-01-08-STE1',
        '2024-08-08-STE1',
    ])
]

TEST_IDS = ['q-1']


def exposition_to_pos(exposition):
    return {
        0.00625: 1,
        0.0125: 2,
        0.025: 3,
        0.05: 4,
        0.1: 5,
        1: 6,
        10: 7,
    }[exposition]


# INCLUDE CORE RULES

include: "generic/_core_rules.smk"
include: "generic/_measures.smk"
include: "generic/_join_results.smk"


# RULES

rule fig_SX:
    input:
        measures_all='figures/data/figSX/measures_all_q1.csv'
    output:
        svg='figures/panels/figSX.svg',
        png='figures/panels/figSX.png',
    resources:
        mem_gib=1
    run:
        measures_all = pd.read_csv(input.measures_all)

        fig, ax = subplots_from_axsize(axsize=(1.4, 6.), top=0.4, right=.6, left=.6, bottom=.8)

        measures_all['pos'] = measures_all['exposition'].apply(exposition_to_pos)

        jax_plots.plot_mi(
            ax, 
            measures_all, 
            field='first_pulse_response',
            title='response amplitude \n to single pulse\n[log fold change]',
            plot_labels=False,
            annotate_means=False,
            marker='o',
            clip_on=False,
        )

        ax.set_ylim(7.5, .5)
        ax.set_yticks(*list(zip(*[(exposition_to_pos(exposition), f"{exposition} s") for exposition in measures_all['exposition'].unique()])))
        ax.axvline(0, ls='-', alpha=.1, color='k')

        fig.savefig(str(output.svg))
        fig.savefig(str(output.png), dpi=300)



rule fig_S3_average:
    input:
        measures_all='figures/data/figS3/measures_all_q-1.csv'
    output:
        svg='figures/panels/figS3_average.svg',
        png='figures/panels/figS3_average.png',
    resources:
        mem_gib=1
    run:
        measures_all = pd.read_csv(input.measures_all)

        fig, ax = subplots_from_axsize(axsize=(1.4, 6.), top=0.4, right=.6, left=.6, bottom=.8)

        measures_all['pos'] = measures_all['exposition'].apply(exposition_to_pos)

        jax_plots.plot_mi(
            ax, 
            measures_all, 
            field='average_response_amplitude',
            title='average response amplitude\n[log fold change]',
            plot_labels=False,
            annotate_means=False,
            marker='o',
            clip_on=False,
        )

        ax.axvline(0, ls='-', alpha=.1, color='k')
        ax.set_ylim(7.5, .5)
        ax.set_yticks(*list(zip(*[(exposition_to_pos(exposition), f"{exposition} s") for exposition in measures_all['exposition'].unique()])))

        fig.savefig(str(output.svg))
        fig.savefig(str(output.png), dpi=300)


