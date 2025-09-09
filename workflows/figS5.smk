import matplotlib
from subplots_from_axsize import subplots_from_axsize
import pandas as pd
import seaborn as sns

from generic._defaults import (
    DATA_MANAGER,
    WELLS_SELECTED,
    SET_TYPES,
    MODEL_IDS,
    DATASET_IDS,
    TRAIN_IDS,
    TEST_IDS,
)

from src import fig_style
from matplotlib.ticker import MultipleLocator
from config.local_config import OUTPUT_PATH

matplotlib.use("agg")
workdir: OUTPUT_PATH


# CONFIGS

TEST_IDS = ['q0']

dataset_id = DATASET_IDS[0]
model_id = MODEL_IDS[0]
train_id = TRAIN_IDS[0]
test_id = TEST_IDS[0]

set_id = 'main+STE1+criz'

# INCLUDE CORE RULES

include: "generic/_core_rules.smk"
include: "generic/_evaluate_network.smk"

# RULES

rule fig_S5:
    input:
        f'cache/network_evaluation/single/per_set/{set_id}/li-mean-p-network-logit_{dataset_id}_{model_id}_{train_id}_{test_id}.csv'
    output:
        svg='figures/panels/figS5.svg',
        png='figures/panels/figS5.png',
    run:
        fig, (ax, cbar_ax) = subplots_from_axsize(axsize=([2, .1], 2), wspace=.2, right=.6)

        li_mean_p_network = pd.read_csv(str(input), index_col='L')

        li_mean_p_network.index.name = '$last_k$ [min]'
        li_mean_p_network.columns.name = '$interval_k$ [min]'

        # sns.heatmap(li_mean_mlp_logit, center=0, square=True, vmin=-3.5, vmax=3.5, ax=ax, cbar_ax=cbar_ax)
        # xticks = [x for x in li_mean_p_network.columns if not x % 5]
        # yticks = [y for x in li_mean_p_network.columns if not x % 5]
        sns.heatmap(
            li_mean_p_network,
            center=0,
            square=True,
            vmin=-2.1,
            vmax=2.1,
            ax=ax,
            cbar_ax=cbar_ax,
            cbar_kws=dict(
                label='mean logit update $u_{Bayes}$',
                ticks=MultipleLocator(1),
            ),
            xticklabels=5,
            yticklabels=5,
        )

        ax.set_axisbelow(True)
        ax.grid(color='k', linestyle=':', alpha=.2)

        # ax.yaxis.set_major_locator(MultipleLocator(5))

        fig.savefig(str(output.svg))
        fig.savefig(str(output.png), dpi=300)


