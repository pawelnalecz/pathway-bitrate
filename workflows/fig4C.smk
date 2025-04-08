import numpy as np
import pandas as pd
import matplotlib
from subplots_from_axsize import subplots_from_axsize

from generic._defaults import (
    DATA_MANAGER,
    WELLS_SELECTED,
    SET_TYPES,
)

from src import jax_protocol
from src import fig_style
from config.local_config import OUTPUT_PATH

matplotlib.use("agg")
workdir: OUTPUT_PATH


# CONFIGS

dataset_id = 'ls+cell+inhs'
model_id = 'nn'
train_id = 'main-q0+opt-5-90protocol-L2-003-q1tr'
# test_id = 'q1-reweight-optprotocol'

DATASET_IDS = [dataset_id]
MODEL_IDS = [model_id]
TRAIN_IDS = [train_id]
# TEST_IDS = [test_id]


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

# RULES

rule fig_4C:
    input:        
        expand(
            f'cache/train/main/{{optset_id}}/protocol_{dataset_id}_{model_id}_{train_id}.yaml',
            optset_id=[set_type for set_type, _, in set_types_and_colors]
        )
    output:
        svg='figures/panels/fig4C.svg',
        png='figures/panels/fig4C.png',
    run:
        fig, ax = subplots_from_axsize(
            axsize=(6, 1.8),
            # axsize=(6, 1.3),
            ncols=1,
            left=.65,
            top=.3,
            facecolor=(1., .95, .9),
        )
        xmin = 5
        xmax = 90
        experimental_protocol = DATA_MANAGER.predefined_protocols['long_experimental']
        ax.plot(np.arange(xmin, xmax + 1), pd.Series(np.exp(experimental_protocol.interval_logprobs)).reindex(np.arange(xmin - 1, xmax), fill_value=0), marker='o', ms=3, ls='-', label='experimental', color='k', alpha=.5)
        # ax.bar(experimental_protocol.ls[xmin-1:] / 60, np.exp(experimental_protocol.interval_logprobs)[xmin-1:], ls='-', color='k', label='experimental', fill=False)

        for protocol_yaml, (set_type, color) in zip(list(input), set_types_and_colors):
            protocol = jax_protocol.Protocol(path=protocol_yaml)
            ax.plot(protocol.ls[xmin-1:] / 60, np.exp(protocol.interval_logprobs)[xmin-1:], marker='o', ms=3, ls='-', label=set_type.replace('main+', ''), color=color, alpha=.5)
        # ax.plot(experimental_protocol.ls[xmin-1:] / 60, np.exp(experimental_protocol.interval_logprobs)[xmin-1:], ls='-', color='k')
        # ax.legend()
        ax.set_xlabel('interval between pulses [min]')
        ax.set_ylabel('probability')
        ax.spines[['top', 'right']].set_visible(False)

        # ax.set_yscale('log')

        fig.savefig(str(output.svg))
        fig.savefig(str(output.png), dpi=300)
