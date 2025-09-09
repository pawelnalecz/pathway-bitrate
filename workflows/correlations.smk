import matplotlib
from subplots_from_axsize import subplots_from_axsize
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from generic._defaults import (
    DATA_MANAGER,
    WELLS_SELECTED,
    DATASET_IDS,
    MODEL_IDS,
    TRAIN_IDS,
    TEST_IDS,
    SET_TYPES,
)

from config import parameters
from config.local_config import OUTPUT_PATH

matplotlib.use("agg")
workdir: OUTPUT_PATH



# INCLUDE CORE RULES

include: "generic/_combine_plots.smk"
include: "generic/_core_rules.smk"
include: "generic/_neighbors.smk"

# RULES

rule all:
    input:
        expand(
            'correlations/correlations_{dataset_id}_{model_id}_{train_id}_{test_id}.png',
            dataset_id=DATASET_IDS,
            model_id=MODEL_IDS,
            train_id=TRAIN_IDS,
            test_id=TEST_IDS,
        )


rule correlations:
    input:
        **{
            "tracks_mi": 'cache/tracks_mi/{per}/{well_or_set_id}/tracks_mi_{dataset_id}_{model_id}_{train_id}_{test_id}.csv.gz',
            "tracks_info": 'cache/preprocessed/{per}/{well_or_set_id}/tracks_info.csv.gz',
            "track_confluence": 'cache/neighbors/{per}/{well_or_set_id}/track_confluence.csv.gz',
        }
    output:
        'correlations/single/{per}/{well_or_set_id}/correlations_{dataset_id}_{model_id}_{train_id}_{test_id}.csv',
    resources:
        mem_mib=512
    run:

        index_col = 'track_id' if wildcards.per == 'per_well' else ['well_id', 'track_id']
        tracks_info = pd.read_csv(input.tracks_info, index_col=index_col)
        tracks_mi = pd.read_csv(input.tracks_mi, index_col=index_col)
        track_confluence = pd.read_csv(input.track_confluence, index_col=index_col)
        tracks_mi_and_info = tracks_mi.join(tracks_info).join(track_confluence)

        tracks_mi_and_info.columns.name = 'features'

        corrs = tracks_mi_and_info.corr().dropna(how='all').dropna(axis='columns', how='all')

        corrs.to_csv(str(output))


rule plot_correlations:
    input:
        **{
            well_id: f'correlations/single/per_well/{well_id}/correlations_{{dataset_id}}_{{model_id}}_{{train_id}}_{{test_id}}.csv'
            for well_id in WELLS_SELECTED.index
        },
    output:
        png='correlations/correlations_{dataset_id}_{model_id}_{train_id}_{test_id}.png',
        svg='correlations/correlations_{dataset_id}_{model_id}_{train_id}_{test_id}.svg',
    resources:
        mem_mib=512
    run:
        correlations = pd.concat(
            (
                pd.read_csv(input[well_id], index_col='features')
                for well_id in WELLS_SELECTED.index
            ),
            names=['well_id'],
            keys=WELLS_SELECTED.index,
        )

        print(correlations)
        corrs_reshaped = correlations['mi_cross_per_slot'].unstack('well_id')
        print(corrs_reshaped)

        fig, ax = subplots_from_axsize(axsize=tuple(0.2 * np.array(corrs_reshaped.shape)), left=3, bottom=3, right=2)

        ax.imshow(corrs_reshaped.to_numpy(), cmap='RdBu_r', vmin=-1, vmax=1.)
        ax.set_xticks(range(len(corrs_reshaped.columns)), corrs_reshaped.columns, rotation=70, horizontalalignment='right', verticalalignment='top')
        ax.set_yticks(range(len(corrs_reshaped.index)), corrs_reshaped.index)

        ax2 = ax.twinx()

        ax2.set_ylim(len(corrs_reshaped.index) + .5, -1.5)
        ax2.set_yticks(range(len(corrs_reshaped.index)), corrs_reshaped.mean(axis=1).apply(lambda x: f"{x:.2f}"))


        fig.savefig(str(output.svg))
        fig.savefig(str(output.png), dpi=300)
