import pandas as pd
import matplotlib
from subplots_from_axsize import subplots_from_axsize

from generic._defaults import (
    DATA_MANAGER,
    WELLS_SELECTED,
    SET_TYPES,
)

from src import jax_plots
from src import fig_style
from config.parameters import RESPONSIVENESS_DELAY
from config.local_config import OUTPUT_PATH

matplotlib.use("agg")
workdir: OUTPUT_PATH


# CONFIGS

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
include: "generic/_measures.smk"

# RULES

rule fig_4A:
    input:
        response_over_reference=expand(
            f'cache/response_amplitude/single/per_set/{{set_id}}/response-over-reference_q1_{RESPONSIVENESS_DELAY}.csv',
            set_id=[set_id for set_id, _ in set_types_and_colors]
        )
    output:
        svg='figures/panels/fig6B.svg',
        png='figures/panels/fig6B.png',
    resources:
        mem_gib=1
    run:
        fig, ax = subplots_from_axsize(axsize=(2., 1.), top=.3)

        for input_path, (set_id, color) in zip(input.response_over_reference, set_types_and_colors):
            response_over_reference = pd.read_csv(input_path, index_col='L')['response_amplitude']
            jax_plots.plot_log_response_over_reference_by_interval(ax, response_over_reference, color=color, marker='o', ms=2)
        
        ax.set_xlim(-.5, 35.5)
        ax.set_xlabel('interval between pulses [min]')
        ax.set_title(f'{RESPONSIVENESS_DELAY} min after pulse')
        ax.spines[['top', 'right']].set_visible(False)

        fig.savefig(str(output.svg))
        fig.savefig(str(output.png), dpi=300)


