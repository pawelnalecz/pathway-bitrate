import matplotlib
from subplots_from_axsize import subplots_from_axsize
import pandas as pd
import json
import re


from generic._defaults import (
    DATA_MANAGER,
    WELLS_SELECTED,
    SET_TYPES,
    DATASET_IDS,
    MODEL_IDS,
    TRAIN_IDS,
    TEST_IDS,
)

# from config.configs import (
#     DATASET_CONFIGS,
# )


from config.local_config import OUTPUT_PATH

matplotlib.use("agg")
workdir: OUTPUT_PATH


# INCLUDE CORE RULES

include: "generic/_combine_plots.smk"
include: "generic/_core_rules.smk"


# CONFIGS


SLICE_START_RANGE = [1] #range(-1, 4)
SLICE_END_RANGE = range(1, 13)
MAX_SLICE_LENGTH = 12

SLICES = {
    f'r{slice_start}-r{slice_end}': {
         'r_ts': list(range(60*slice_start, 60*(slice_end + 1), 60)),
    }
    for slice_start in SLICE_START_RANGE
    for slice_end in SLICE_END_RANGE
    if 0 < slice_end - slice_start and slice_end - slice_start < MAX_SLICE_LENGTH
}

DATASET_CONFIGS = {
    f'{dataset_id}-{slice_id}': DATASET_CONFIGS[dataset_id] | slice_config
    for slice_id, slice_config in SLICES.items() 
    for dataset_id in DATASET_IDS
}

DATASET_IDS_ORIGINAL = DATASET_IDS

DATASET_IDS = [
    f'{dataset_id}-{slice_id}'
    for slice_id in SLICES.keys() 
    for dataset_id in DATASET_IDS
]

print(DATASET_CONFIGS)
print(DATASET_IDS)


# INCLUDE JOINING (must go after DATASET_CONFIGS are redefined)

include: "generic/_join_results.smk"

# RULES

rule all:
    input:
        expand(
            'plot_slice_scan/combined/per_experiment/{experiment}/slice-scan_{dataset_id}_{model_id}_{train_id}_{test_id}.png',
            experiment=WELLS_SELECTED['experiment'].unique(),
            dataset_id=DATASET_IDS_ORIGINAL,
            model_id=MODEL_IDS,
            train_id=TRAIN_IDS,
            test_id=TEST_IDS,
        )


rule plot_slice_scan:
    input:
        'plot_slice_scan/mi_all.csv'
    output:
        'plot_slice_scan/single/per_well/{well_id}/slice-scan_{dataset_id}_{model_id}_{train_id}_{test_id}.png'
    resources:
        mem_mib=128
    run:

        mi_all = pd.read_csv(str(input))

        expr = re.compile(r"(.+)-r\-?[0-9]+\-r\-?[0-9]+")
        
        train_ids = wildcards.train_id.split('+')
        max_train_ids = len([col for col in mi_all.columns if col.startswith('train_id_')])
       
        mi_all = mi_all[
            (mi_all['well_id'] == wildcards.well_id)
          & (mi_all['dataset_id'].apply(lambda x: expr.match(x).group(1)) == wildcards.dataset_id)
          & (mi_all['model_id'] == wildcards.model_id)
          & (mi_all[[f'train_id_{it}' for it in range(len(train_ids))]] == train_ids).all(axis=1)
          & (mi_all[[f'train_id_{it}' for it in range(len(train_ids), max_train_ids)]].isna()).all(axis=1)
          & (mi_all['test_id'] == wildcards.test_id)
        ]

        mi_all['slice_start'] = mi_all['r_ts'].apply(lambda x: json.loads(x)[0]  / 60)
        mi_all['slice_end']   = mi_all['r_ts'].apply(lambda x: json.loads(x)[-1] / 60)

        fig, ax = subplots_from_axsize(axsize=(3,3))
        ax.scatter(
            mi_all['slice_start'],
            mi_all['slice_end'],
            c=mi_all['mi_ce'],
            vmin=-3,
            vmax=10,
            cmap='jet',
        )

        fig.savefig(str(output))
