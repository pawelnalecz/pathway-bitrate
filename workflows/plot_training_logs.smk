import pandas as pd
import matplotlib
from subplots_from_axsize import subplots_from_axsize

from generic._defaults import (
    DATA_MANAGER,
    WELLS_SELECTED,
    DATASET_IDS,
    MODEL_IDS,
    TRAIN_IDS,
)

from config.local_config import OUTPUT_PATH

matplotlib.use("agg")
workdir: OUTPUT_PATH


# CONFIGS

SET_TYPES = [
    'main',
]

# INCLUDE CORE RULES

include: "generic/_core_rules.smk"

# RULES

rule all:
    input:
        expand(
            'plot_training_logs/{trainset_id}/log_{dataset_id}_{model_id}_{train_id}.png',
            trainset_id=SET_ID_TO_WELL_IDS,
            dataset_id=DATASET_IDS,
            model_id=MODEL_IDS,
            train_id=TRAIN_IDS,
        )


rule plot_training_log:
    input:
        training_log='cache/train/{trainset_id}/log_{dataset_id}_{model_id}_{train_id}.csv'
    output:
        'plot_training_logs/{trainset_id}/log_{dataset_id}_{model_id}_{train_id}.png'
    run:
        training_log = pd.read_csv(input.training_log, index_col='step')
        fig, ax = subplots_from_axsize(left=0.7, top=0.5)
        ax.plot(training_log['cMI'], '-k', alpha=0.2)
        ax.plot(training_log['cMI'].rolling(window=50, center=True).mean(), '-k')
        ax.set_xlabel('step')
        ax.set_ylabel('cMI')
        ax.set_title(f"{wildcards.trainset_id}\n{wildcards.dataset_id}; {wildcards.model_id}; {wildcards.train_id}")
        fig.savefig(str(output), dpi=300)


