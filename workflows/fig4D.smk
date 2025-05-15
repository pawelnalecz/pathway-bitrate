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

from src.jax_plots import plot_mi_diff
from src import fig_style
from config.local_config import OUTPUT_PATH

matplotlib.use("agg")
workdir: OUTPUT_PATH

# CONFIGS

train_id_0 = 'main-q0'
train_id_1 = 'opt-5-90protocol-L2-003-q1tr'
test_id = 'q1tr'
optimized_test_id = 'q1tr-reweight-optprotocol'

WELLS_SELECTED = WELLS_SELECTED[
    (
        (WELLS_SELECTED['cell_line'] == 'BEAS2B') 
      & (WELLS_SELECTED['inh_crizotinib'] == 0) 
      & ((WELLS_SELECTED['inh_trametinib'] == 0) | (WELLS_SELECTED['inh_cyclosporin'] == 0))
    ) |
    (
        (WELLS_SELECTED['cell_line'] == 'STE1') 
      & (WELLS_SELECTED['inh_crizotinib'] > 0) 
    )
]

TRAIN_IDS = [train_id_0, f"{train_id_0}+{train_id_1}"]
TEST_IDS = [test_id, optimized_test_id]

# INCLUDE CORE RULES

include: "generic/_core_rules.smk"
include: "generic/_join_results.smk"

# RULES

rule fig_4D:
    input:
        mi_all='figures/data/fig4D/mi_all.csv'
    output:
        svg='figures/panels/fig4D.svg',
        png='figures/panels/fig4D.png',
    resources:
        mem_gib=1
    run:
        index_cols = ['well_id', 'dataset_id', 'model_id', 'train_id_0']
        mi_all = pd.read_csv(input.mi_all).set_index(index_cols)
        
        fig, ax = subplots_from_axsize(
            axsize=(1.4, 3.0), 
            top=0.4, 
            left=.8,
            )

        mi_no_opt = mi_all[mi_all['train_id_1'].isna() & (mi_all['test_id'] == test_id)]
        mi_opt = mi_all[mi_all['train_id_1'].notna() & (mi_all['test_id'] == optimized_test_id)]

        plot_mi_diff(
            ax, 
            mi_no_opt,
            mi_opt, 
            field='mi_ce',
            title='channel capacity [bit/h]',
            plot_labels=True,
            means_format_str="{mean_1:.1f} → {mean_2:.1f} bit/h",
            ymin=-.5,
            ymax=20.,
            marker='o',
            clip_on=False,
        )
        
        fig.savefig(str(output.svg))
        fig.savefig(str(output.png), dpi=300)

