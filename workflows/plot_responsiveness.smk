from pathlib import Path
import sys
sys.path.append(str(Path('.').resolve()))

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
from subplots_from_axsize import subplots_from_axsize
from itertools import product

from generic._defaults import (
    DATA_MANAGER,
    WELLS_SELECTED,
    SET_TYPES,
)

from config.configs import (
    DATASET_CONFIGS,
)

from config.parameters import RESPONSIVENESS_DELAY
from config.local_config import OUTPUT_PATH

matplotlib.use("agg")
workdir: OUTPUT_PATH

# CONFIGS

PLOTS = [
    'responsiveness-heatmap_q0',
    'responsiveness_q0',
    ]


# INCLUDE CORE RULES

include: "generic/_core_rules.smk"
include: "generic/_combine_plots.smk"


# RULES

rule all:
    input:
        expand(
            'plot_responsiveness/combined/per_experiment/{experiment}/{plot}.png',
            experiment=WELLS_SELECTED['experiment'].unique(),
            plot=PLOTS,
        ),
    

rule plot_optimal_protocol:
    input:        
        lambda wildcards: f"cache/preprocessed/per_well/{{well_id}}/tracks_preprocessed{'' if int(wildcards.quality) >= 0 else '_all'}.pkl.gz",
        lambda wildcards: f"cache/preprocessed/per_well/{{well_id}}/tracks_info{'' if int(wildcards.quality) >= 0 else '_all'}.csv.gz",
    output:
        'plot_responsiveness/single/per_well/{well_id}/responsiveness-heatmap_q{quality}.png',
        'plot_responsiveness/single/per_well/{well_id}/responsiveness_q{quality}.png',
    run:
        well_id = wildcards.well_id
        experiment = DATA_MANAGER.get_experiment(well_id)
        pulses = DATA_MANAGER.get_pulses(experiment)['time_in_seconds']

        dataset = load_dataset(well_id, quality=int(wildcards.quality), r_ts=[0, RESPONSIVENESS_DELAY * 60], **DATASET_CONFIGS['raw'], differentiate=False)
        I2_lookup = pd.Series(pulses.diff().tolist(), index=pulses)
        dataset['ratio'] = -dataset['X+0'] + dataset[f'X+{RESPONSIVENESS_DELAY * 60}']
        dataset['no_future_pulse'] = (dataset['I'] - dataset['L']) > RESPONSIVENESS_DELAY * 60
        
        ratio_by_li = np.exp(dataset.groupby(['L', 'I'])['ratio'].mean())
        ratio_background = np.exp(dataset[dataset['no_future_pulse']].groupby('L')['ratio'].mean())
        ratio_by_li_normalized = ratio_by_li / ratio_background

        ls = np.arange(0, ratio_by_li.index.get_level_values('L').max() + 60, 60)
        lis = list(product(ls, repeat=2))
        
        fig, (ax, ax_background, ax_normalized, cbar_ax) = subplots_from_axsize(axsize=([5, .2, 5, .1],5), wspace=[.1, .15, 0.3], top=.6, right=.5)

        vmin = .7
        vmax = 1.3
        scatter = ax.imshow(ratio_by_li.reindex(lis).unstack().to_numpy(), cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
        ax_background.imshow(ratio_background.reindex(ls).to_numpy().reshape(-1, 1), cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
        ax_normalized.imshow(ratio_by_li_normalized.reindex(lis).unstack().to_numpy(), cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
        plt.colorbar(scatter, cax=cbar_ax)

        lmax = ls.max() / 60 + 1
        ax.set_xlim(0, lmax)
        ax.set_ylim(lmax, 0)
        ax.set_xlabel('I')
        ax.set_ylabel('L')
        ax.grid(ls='--', alpha=.3)
        ax.set_title('translocation ratio [6 min : 0 min]')
        
        ax_background.set_ylim(lmax, 0)
        ax_background.set_xticks([0.5])
        ax_background.set_xticklabels(['bg'])
        ax_background.set_yticklabels('')
        ax_background.grid(ls='--', alpha=.3)

        ax_normalized.set_xlim(0, lmax)
        ax_normalized.set_ylim(lmax, 0)
        ax_normalized.set_yticklabels('')
        ax_normalized.set_xlabel('I')
        ax_normalized.grid(ls='--', alpha=.3)
        ax_normalized.set_title('relative change of translocation ratio [6 min : 0 min]')

        fig.suptitle(well_id)
        fig.savefig(str(output[0]))


        fig, axs = subplots_from_axsize(ncols=2, top=.3, wspace=.2)

        axs[0].plot(ls / 60, ratio_by_li.reindex(list(zip(ls, ls))), color='olive', marker='o', ms=3, label='second pulse present')
        axs[0].plot(ls / 60, ratio_background.reindex(ls), color='maroon', marker='o', ms=3, label='second pulse absent')
        axs[0].legend()
        axs[0].set_xlabel('time since previous pulse')

        axs[1].plot(ls / 60, ratio_by_li_normalized.reindex(list(zip(ls, ls))), color='brown', marker='o', ms=3, label='ratio')
        axs[1].legend()
        axs[1].set_xlabel('time since previous pulse')

        fig.suptitle(well_id)
        fig.savefig(str(output[1]))

