from matplotlib import pyplot as plt
import matplotlib
from matplotlib.ticker import MultipleLocator
from subplots_from_axsize import subplots_from_axsize
import pandas as pd
import numpy as np

from generic._defaults import (
    DATA_MANAGER,
    WELLS_SELECTED,
    SET_TYPES,
)

from src import fig_style
from src.fig_layout import set_to_label
from config.local_config import OUTPUT_PATH

matplotlib.use("agg")
workdir: OUTPUT_PATH


# CONFIGS

set_types_and_colors = [
    ('main+STE1+0uM', 'slategray'),
    # ('main+STE1+criz03uM', 'skyblue'),
    # ('main+STE1+criz1uM', 'deepskyblue'),
    # ('main+STE1+criz3uM', 'dodgerblue'),
    ('main+STE1+criz', 'deepskyblue'),

    ('main+BEAS2B+0uM', 'goldenrod'),
    ('main+BEAS2B+cycl1uM', 'gold'),
    ('main+BEAS2B+tram05uM', 'red'),
    # ('main+BEAS2B+criz03uM', 'sandybrown'),
    # ('main+BEAS2B+tram05uMcycl1uM', 'navajowhite'),

]

set_ids = [set_id for set_id, _ in set_types_and_colors]

# INCLUDE CORE RULES

include: "generic/_core_rules.smk"

# RULES

rule fig_6A:
    input:
        mean_trajectories=expand(
            'cache/mean_trajectory/per_set/{set_id}/mean_trajectory_q1.csv.gz',
            set_id=set_ids,
        ),
    output:
        svg='figures/panels/fig6A.svg',
        png='figures/panels/fig6A.png',
    resources:
        mem_mib=512,
    run:
        pulses = DATA_MANAGER.get_pulses(WELLS_SELECTED.iloc[0]['experiment'])['time_in_seconds']

        pulse_id = 33
        xmin = -5 * 60
        xmax = 20 * 60

        # xmin = 50
        # xmax = 250 * 60

        fig, ax = subplots_from_axsize(axsize=(.8, 1.), left=.7, top=.3)
        conditions = []
        for mean_trajectory_path, (set_id, color) in zip(input.mean_trajectories, set_types_and_colors):
            mean_trajectory = pd.read_csv(mean_trajectory_path, index_col='time_in_seconds')
            mean_trajectory.index = mean_trajectory.index - pulses[pulse_id]
            mean_trajectory_normalized = mean_trajectory - mean_trajectory.loc[0]
            mean_trajectory_normalized.plot(
                ax=ax,
                color=color, 
                alpha=.6, 
                label=set_id,
                )
        for pulse in pulses:
            ax.axvline(pulse - pulses[pulse_id], ls='--', alpha=.3, color='k')
            
        
        ax.xaxis.set_major_locator(MultipleLocator(5 * 60))
        ax.xaxis.set_minor_locator(MultipleLocator(60))
        ax.xaxis.set_major_formatter(lambda x, pos: f"{x // 60:.0f}" if not x % 300 else '')
        ax.set_xlim(xmin, xmax)

        ax.yaxis.set_major_locator(MultipleLocator(.2))

        ax.set_ylabel('ERK-KTR trajectory')
        ax.set_xlabel('time after pulse [min]')
        # ax.legend([set_to_label[set_id].replace('+', '').replace('\n', ' + ') for set_id in set_ids], loc=(1.1, 0))
        ax.get_legend().set_visible(False)
        fig.savefig(str(output.svg))
        fig.savefig(str(output.png), dpi=300)

