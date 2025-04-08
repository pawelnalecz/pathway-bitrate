import matplotlib
from subplots_from_axsize import subplots_from_axsize
import pandas as pd
import numpy as np

from generic._defaults import (
    DATA_MANAGER,
    WELLS_SELECTED,
    SET_TYPES,
)

from src.per_track_plots import get_fraction_transmitting
from src.stat_utils import get_sorted
from src import fig_style
from src.fig_layout import set_to_label
from config.local_config import OUTPUT_PATH

matplotlib.use("agg")
workdir: OUTPUT_PATH


# CONFIGS

dataset_id = 'ls+cell+inhs'
train_id = 'main-q0'
test_id = 'q1'
model_id = 'nn'

DATASET_IDS = [dataset_id]
MODEL_IDS = [model_id]
TRAIN_IDS = [train_id]
TEST_IDS = [test_id]

set_ids_with_spaces = [
    'main+STE1+0uM', 
    'main+STE1+criz03uM', 
    'main+STE1+criz1uM', 
    'main+STE1+criz3uM',
    None,
    'main+BEAS2B+0uM', 
    'main+BEAS2B+cycl1uM', 
    'main+BEAS2B+tram05uM', 
    # 'main+BEAS2B+criz03uM', 
    # 'main+BEAS2B+tram05uMcycl1uM', 
]

set_ids = [set_id for set_id in set_ids_with_spaces if set_id]

# INCLUDE CORE RULES

include: "generic/_core_rules.smk"
include: "generic/_combine_plots.smk"

# RULES

rule fig_3A:
    input:
        expand(
            'cache/preprocessed/per_set/{set_id}/tracks_info.csv.gz',
            set_id=set_ids
        ),
        expand(
            f'cache/tracks_mi/per_set/{{set_id}}/tracks_mi_{dataset_id}_{model_id}_{train_id}_{test_id}.csv.gz',
            set_id=set_ids
        )
    output:
        svg='figures/panels/fig3A.svg',
        png='figures/panels/fig3A.png',
    run:
        ax_height = 5 / 9.5 
        hspace_default = 1 / 9.5 
        hspaces = []
        hspace = hspace_default
        for set_id in set_ids_with_spaces:
            if set_id:
                hspaces.append(hspace)
                hspace = hspace_default
            else:
                hspace += .5 * (ax_height + hspace_default)
        hspaces.pop(0)

        fig, axs = subplots_from_axsize(nrows=len(set_ids), axsize=(2.4, ax_height), hspace=hspaces, top=0.4 + hspace_default / 2, bottom=.8 + hspace_default / 2, left=.9)
        for set_id, ax in zip(set_ids, axs):
            well_ids = SET_ID_TO_WELL_IDS[set_id]
            experiments = [DATA_MANAGER.get_experiment(well_id) for well_id in  well_ids]
            seconds_per_timepoint = DATA_MANAGER.get_seconds_per_timepoint_for_experiment_list(experiments)
            starts_ends = [DATA_MANAGER.get_effective_time_range(experiment) for experiment in experiments]
            starts, ends = list(zip(*starts_ends))
            start = min(starts)
            end = max(ends)
            
            bph = 60 * 60 / seconds_per_timepoint / np.log(2)
            experiment_length = (end - start) // seconds_per_timepoint

            protocol = DATA_MANAGER.predefined_protocols['long_experimental']
            pe = protocol.entropy_per_slot()

            index_col = ['well_id', 'track_id']
            tracks_mi = pd.read_csv(f'cache/tracks_mi/per_set/{set_id}/tracks_mi_{dataset_id}_{model_id}_{train_id}_{test_id}.csv.gz', index_col=index_col)
            tracks_info = pd.read_csv(f'cache/preprocessed/per_set/{set_id}/tracks_info.csv.gz', index_col=index_col)
            tracks_mi_and_info = tracks_mi.join(tracks_info).dropna()

            
            by = 'mi_cross_per_slot'
            field = 'mi_cross_per_slot'

            field_sorted, slots_sorted = get_sorted(tracks_mi_and_info, field, by)
            average, fraction_transmitting, average_transmitting, transmitting_thr = get_fraction_transmitting(field_sorted, slots_sorted)

            ax.hist(
                tracks_mi_and_info[by] * bph,
                weights=tracks_mi_and_info['slots'],
                density=True,
                color='k',
                bins=np.linspace(-.4 * pe * bph, 1.2 * pe * bph, 48),
            )


            ax.axvline(average * bph, color='goldenrod', ls=':', label='average')
            ax.axvline(average_transmitting * bph, color='g', ls=':', label='average in transmitting')
            ax.axvline(0 * bph, color='red', ls=':', label='average in non-transmitting')
            ax.axvline(pe * bph, color='grey', ls='-', alpha=.8, lw=.5, label='input information rate')
            ax.axvline(transmitting_thr * bph, color='b', ls='-', lw=.5, label='transmitting threshold')
            # ax.legend()
            ylim = (0, 1.)
            xlim = (-5, 18)
            ax.fill_betweenx(ylim, xlim[0], transmitting_thr * bph, color='r', alpha=.1)
            ax.fill_betweenx(ylim, transmitting_thr * bph, xlim[1], color='g', alpha=.1)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            tpt_all = tracks_mi_and_info['slots'].sum() / experiment_length
    
            # ax.annotate(
            #     f"non-transmitting\n"
            #     f"{1 - fraction_transmitting:.0%} cells\n"
            #     f"bitrate {0:.1f} bit/h",
            #     (0.04, 0.6), xycoords='axes fraction', horizontalalignment='left', verticalalignment='center', color='r')
            
            # ax.annotate(
            #     f"transmitting\n"
            #     f"{fraction_transmitting:.0%} cells\n"
            #     f"bitrate {average_transmitting * bph:.1f} bit/h",
            #     (0.54, 0.6), xycoords='axes fraction', horizontalalignment='left', verticalalignment='center', color='g')
        
            # ax.annotate(
                # f"average bitrate: {average * bph:.1f} bit/h",
                # (0.35, 0.9), xycoords='axes fraction', horizontalalignment='center', verticalalignment='center', color='goldenrod')
            

            ax.set_ylabel(set_to_label[set_id], rotation=0, horizontalalignment='right', verticalalignment='center', fontsize='large')
            ax.set_yticks([])
        
        for ax in axs[:-1]:
            ax.set_xticklabels([])

        axs[-1].set_xlabel('bitrate [bit/h]')
        
        axs[0].annotate(
            '\nnon-transmitting',
            (0.4, .97), xycoords='figure fraction', 
            horizontalalignment='right', verticalalignment='top', color='r')

        
        axs[0].annotate(
            'all cells',
            (0.42, .97), xycoords='figure fraction', 
            horizontalalignment='center', verticalalignment='top', color='goldenrod')

        axs[0].annotate(
            '\ntransmitting',
            (0.44, .97), xycoords='figure fraction', 
            horizontalalignment='left', verticalalignment='top', color='g')

        axs[0].annotate(
            'input information\nrate',
            (.91, .97), xycoords='figure fraction', 
            horizontalalignment='right', verticalalignment='top', color='grey')

        fig.savefig(str(output.svg))
        fig.savefig(str(output.png), dpi=300)

