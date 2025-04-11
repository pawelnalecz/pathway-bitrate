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
from config.local_config import OUTPUT_PATH

matplotlib.use("agg")
workdir: OUTPUT_PATH

# INCLUDE CORE RULES

include: "generic/_plot_mi.smk"
include: "generic/_core_rules.smk"
include: "generic/_join_results.smk"

# RULES

rule all:
    input:
        'mi_defaults/mi_all.csv',
        expand(
            'mi_defaults/mi_{dataset_id}_{model_id}_{train_id}_{test_id}.png',
            well_id=WELLS_SELECTED.index.get_level_values('well_id'),
            dataset_id=DATASET_IDS,
            model_id=MODEL_IDS,
            train_id=TRAIN_IDS,
            test_id=TEST_IDS,
        ),

