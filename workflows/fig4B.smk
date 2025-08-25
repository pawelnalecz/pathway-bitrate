from pathlib import Path

import pandas as pd
import matplotlib
from subplots_from_axsize import subplots_from_axsize

from generic._defaults import (
    DATA_MANAGER,
    WELLS_SELECTED,
    SET_TYPES,
)

from src import fig_style

from config.local_config import OUTPUT_PATH

matplotlib.use("agg")
workdir: OUTPUT_PATH


# CONFIGS

dataset_id = 'ls+cell+inhs'
model_id = 'nn'
train_id = 'main-q0'
test_id = 'q1tr'

DATASET_IDS = [dataset_id]
MODEL_IDS = [model_id]
TRAIN_IDS = [train_id]
TEST_IDS = [test_id]


set_types_and_colors = [
    # ('main+STE1+0uM', 'slategray'),
    ('main+STE1+criz', 'deepskyblue'),
    
    ('main+BEAS2B+0uM', 'goldenrod'),
    ('main+BEAS2B+cycl1uM', 'gold'),
    ('main+BEAS2B+tram05uM', 'red'),
    # ('main+BEAS2B+criz03uM', 'sandybrown'),
    # ('main+BEAS2B+tram05uMcycl1uM', 'navajowhite'),
]



# INCLUDE CORE RULES

include: "generic/_core_rules.smk"
include: "generic/_evaluate_network.smk"

# RULES

rule fig_4B:
    input:
        **{
            set_id: f'cache/network_evaluation/single/per_set/{set_id}/li-mean-p-network-logit_{dataset_id}_{model_id}_{train_id}_{test_id}.csv'
            for set_id, _ in set_types_and_colors
        },
    output:
        svg='figures/panels/fig4B.svg',
        png='figures/panels/fig4B.png',
    resources:
        mem_gib=1
    run:
        fig, ax = subplots_from_axsize(axsize=(2., 1.), top=.3)

        for set_id, color in set_types_and_colors:
            li_mean_p_network_logit = pd.read_csv(str(input[set_id]), index_col='L')
            li_mean_p_network_logit.columns = map(int, li_mean_p_network_logit.columns)
            intervals = li_mean_p_network_logit.columns
            assert all(i in li_mean_p_network_logit.index for i in intervals), "Not all intervals found as L"

            li_mean_p_network_logit_diagonal = pd.Series([li_mean_p_network_logit.loc[i, i] for i in intervals], index=intervals)

            ax.plot(intervals, li_mean_p_network_logit_diagonal, color=color, marker='o', ms=2)

            # ax.set_ylim(0, 1)

        ax.grid(color='k', alpha=0.5, ls=':')
        ax.set_xlim(-.5, 35.5)
        ax.set_xlabel('interval between pulses [min]')
        ax.set_ylabel('logit Bayesian update')
        ax.spines[['top', 'right']].set_visible(False)

        fig.savefig(str(output.svg))
        fig.savefig(str(output.png), dpi=300)



rule fig_S5:
    input:
        **{
            set_id: f'cache/network_evaluation/single/per_set/{set_id}/li-mean-p-network-logit_{dataset_id}_{model_id}_{train_id}_{test_id}.csv'
            for set_id, _ in set_types_and_colors
        },
    output:
        svg='figures/panels/figS5.svg',
        png='figures/panels/figS5.png',
    resources:
        mem_gib=1
    run:
        fig, ax = subplots_from_axsize(axsize=(2.4,1.), facecolor=(.9, .9, 1.), top=.3)

        for set_id, color in set_types_and_colors:
            li_mean_p_network_logit = pd.read_csv(str(input[set_id]), index_col='L')
            li_mean_p_network_logit.columns = map(int, li_mean_p_network_logit.columns)
            intervals = li_mean_p_network_logit.columns
            assert all(i in li_mean_p_network_logit.index for i in intervals), "Not all intervals found as L"
            li_mean_p_network_logit = li_mean_p_network_logit.stack()
            li_mean_p_network_logit['tbp'] = li_mean_p_network_logit.index.get_level_values('L') - li_mean_p_network_logit.index.get_level_values('I')
            tbp_mean_p_network = li_mean_p_network_logit.groupby('tbp').mean()

            # sns.heatmap(li_mean_mlp_logit, center=0, square=True, vmin=-3.5, vmax=3.5, ax=ax, cbar_ax=cbar_ax)
            tbp_mean_p_network.plot(ax=ax, color=color, marker='o', ms=2)

            # ax.set_ylim(0, 1)

        ax.grid(color='k', alpha=0.5, ls=':')
        # ax.set_xlim(-.5, 35.5)
        ax.set_xlabel('interval between pulses [min]')
        ax.set_ylabel('logit Bayesian update')
        ax.spines[['top', 'right']].set_visible(False)

        fig.savefig(str(output.svg))
        fig.savefig(str(output.png), dpi=300)
