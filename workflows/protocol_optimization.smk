from pathlib import Path
import sys
sys.path.append(str(Path('.').resolve()))

import numpy as np
import matplotlib
from subplots_from_axsize import subplots_from_axsize
from src import jax_protocol

from generic._defaults import (
    DATA_MANAGER,
    WELLS_SELECTED,
    SET_TYPES,
    DATASET_IDS,
    MODEL_IDS,
    TEST_IDS,
)

from config.local_config import OUTPUT_PATH

matplotlib.use("agg")
workdir: OUTPUT_PATH


# CONFIGS

TRAIN_IDS = [
    'main-q0+opt-long-protocol-L1', 
    'main-q0+opt-long-protocol-L2',
    ]

TEST_SET_TYPES = ['main+cell+inh']

# INCLUDE CORE RULES

include: "generic/_core_rules.smk"
include: "generic/_combine_plots.smk"
include: "generic/_evaluate_network.smk"
include: "generic/_plot_mi.smk"


# RULES

rule all:
    input:
        'protocol_optimization/mi_all.csv',
        expand(
            'protocol_optimization/combined/per_experiment/{experiment}/{plot}_{dataset_id}_{model_id}_{train_id}_{test_id}.png',
            plot=[
                'nn_predictions_heatmap',
            ],
            experiment=WELLS_SELECTED['experiment'].unique(),
            dataset_id=DATASET_IDS,
            model_id=MODEL_IDS,
            train_id=TRAIN_IDS,
            test_id=TEST_IDS,
        ),
        expand(
            'protocol_optimization/combined/per_set_type/{set_type}/{plot_type}_{dataset_id}_{model_id}_{train_id}_{test_id}.png',
            plot_type=[
                'nn_predictions_heatmap',
            ],
            set_type=TEST_SET_TYPES,
            dataset_id=DATASET_IDS,
            model_id=MODEL_IDS,
            train_id=TRAIN_IDS,
            test_id=TEST_IDS,
        ),
        expand(
            'protocol_optimization/mi_{dataset_id}_{model_id}_{train_id}_{test_id}.png',
            well_id=WELLS_SELECTED.index.get_level_values('well_id'),
            dataset_id=DATASET_IDS,
            model_id=MODEL_IDS,
            train_id=TRAIN_IDS,
            test_id=TEST_IDS,
        )

rule plot_optimal_protocol:
    input:        
        protocol='cache/protocol_optimization/{trainset_id}/{optset_id}/protocol_{dataset_id}_{model_id}_{train_id}.yaml'
    output:
        'protocol_optimization/single/per_set/{optset_id}/optimized-protocol_{dataset_id}_{trainset_id}_{model_id}_{protocol_id}.png'
    run:
        protocol = jax_protocol.Protocol(path=input.protocol)

        fig, ax = subplots_from_axsize(
            axsize=(8, 3),
            ncols=1,
            top=.5,
        )

        ax.plot(protocol.ls, np.exp(protocol.interval_logprobs), marker='o', ls='-', color='k', alpha=.5)
        ax.set_title(f'{wildcards.dataset_id} {wildcards.model_id} {wildcards.train_id}')
        fig.savefig(str(output))

