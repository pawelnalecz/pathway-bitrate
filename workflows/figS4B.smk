import pandas as pd
import matplotlib
from subplots_from_axsize import subplots_from_axsize
from matplotlib.ticker import MultipleLocator

from generic._defaults import (
    DATA_MANAGER,
    WELLS_SELECTED,
    SET_TYPES,
)

from src import jax_plots
from src import fig_style
from src.fig_layout import set_to_label
from config.local_config import OUTPUT_PATH

matplotlib.use("agg")
workdir: OUTPUT_PATH


# CONFIGS

delays_limits = [
    (7, -.2, .7),
    (2, -.2, .3),
    ]

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

rule fig_6BC:
    input:
        **{
            f'response_over_reference_{delay}': expand(
                f'cache/response_amplitude/single/per_set/{{set_id}}/response-over-reference_q1_{delay}.csv',
                set_id=[set_id for set_id, _ in set_types_and_colors],
                )
        for delay, *_ in delays_limits
        }

    output:
        svg='figures/panels/fig6BC.svg',
        png='figures/panels/fig6BC.png',
    resources:
        mem_gib=1
    run:
        fig, axs = subplots_from_axsize(ncols=len(delays_limits), axsize=(2., 1.), top=.3, wspace=.4)

        for ax, (delay, ylim1, ylim2) in zip(axs, delays_limits):
            for input_path, (set_id, color) in zip(input[f'response_over_reference_{delay}'], set_types_and_colors):
                response_over_reference = pd.read_csv(input_path, index_col='L')['response_amplitude']
                jax_plots.plot_log_response_over_reference_by_interval(ax, response_over_reference, color=color, marker='o', ms=2, 
                                                                    #    label=set_to_label[set_id]#.replace('\n', ' + ').replace(' + no inh', ''),
                                                                    )
            
            ax.set_xlim(-.5, 35.5)
            ax.set_ylim(ylim1, ylim2)
            ax.set_xlabel('interval between pulses [min]')
            ax.set_title(f'{delay} min after pulse')
            ax.spines[['top', 'right']].set_visible(False)
            ax.yaxis.set_major_locator(MultipleLocator(.2))
            # ax.legend(loc='upper left')


            # fig.savefig(str(output.svg).replace('.svg', '--with_legend.svg'))
            # fig.savefig(str(output.png).replace('.png', '--with_legend.png'), dpi=300)

            # ax.get_legend().set_visible(False)
            
        for ax in axs[1:]:
            ax.set_ylabel('')

        fig.savefig(str(output.svg))
        fig.savefig(str(output.png), dpi=300)

