from generic._defaults import (
    DATA_MANAGER,
    WELLS_SELECTED,
    MODEL_IDS,
    TRAIN_IDS,
    TEST_IDS,
)


from config.configs import (
    DATASET_CONFIGS,
)

from config.local_config import OUTPUT_PATH

workdir: OUTPUT_PATH

# CONFIGS

DATASET_CONFIGS = DATASET_CONFIGS | {
    'ls': dict(extra_cols=['X_ratio_std']),
    'lsi': dict(extra_cols=['X_ratio_std', 'X_inhibitor']),
}

DATASET_IDS = ['ls', 'lsi']

TRAIN_ONS = [
    'exp-self',
    'exp+inh-self',
]


# INCLUDE CORE RULES

include: "generic/_core_rules.smk"
include: "generic/_join_results.smk"

# RULES

rule all:
    input:
        'mi_X_inhibitor/mi_all.csv'

