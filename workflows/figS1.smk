import matplotlib
from subplots_from_axsize import subplots_from_axsize
import pandas as pd
import numpy as np
from matplotlib.ticker import MultipleLocator

from generic._defaults import (
    DATA_MANAGER,
    WELLS_SELECTED,
    SET_TYPES,
)

from src.fig_layout import set_to_label
from src import fig_style
from src.per_track_plots import plot_hist_with_rolling
from config.local_config import OUTPUT_PATH

matplotlib.use("agg")
workdir: OUTPUT_PATH


# CONFIGS

dataset_id = 'ls+cell+inhs'
train_id = 'main-q0'
test_id = 'q0'
model_id = 'nn'

DATASET_IDS = [dataset_id]
MODEL_IDS = [model_id]
TRAIN_IDS = [train_id]
TEST_IDS = [test_id]

set_type = 'main+cell+inh'

# INCLUDE CORE RULES

include: "generic/_core_rules.smk"
include: "generic/_combine_plots.smk"

set_ids_layout = np.array([
    ['main+STE1+0uM',      'main+BEAS2B+0uM'            ],
    ['main+STE1+criz03uM', 'main+BEAS2B+cycl1uM'        ],
    ['main+STE1+criz1uM',  'main+BEAS2B+tram05uM'       ],
    ['main+STE1+criz3uM',  'main+BEAS2B+criz03uM'       ],
    # [None,                 'main+BEAS2B+tram05uMcycl1uM'],
])
set_ids = [set_id for set_id in set_ids_layout.flatten() if set_id]

# RULES

rule fig_S1:
    input:
        **{
            f"tracks_info_{set_id}": f'cache/preprocessed/per_set/{set_id}/tracks_info.csv.gz'
            for set_id in set_ids
        }
        | {
            f"tracks_mi_{set_id}": f'cache/tracks_mi/per_set/{set_id}/tracks_mi_{dataset_id}_{model_id}_{train_id}_{test_id}.csv.gz'
            for set_id in set_ids
        }
    output:
        svg='figures/panels/figS1.svg',
        png='figures/panels/figS1.png',
    run:
        fig, axs = subplots_from_axsize(*set_ids_layout.shape, axsize=(1.5, 1.), top=.3, wspace=.3)

        for ax, set_id in zip(axs.flatten(), set_ids_layout.flatten()):
            if set_id is None:
                ax.set_visible(False)
                continue
            well_ids = SET_ID_TO_WELL_IDS[set_id]

            index_col = ['well_id', 'track_id']
            tracks_mi = pd.read_csv(input[f"tracks_mi_{set_id}"], index_col=index_col)
            tracks_info = pd.read_csv(input[f"tracks_info_{set_id}"], index_col=index_col)
            tracks_mi_and_info = tracks_mi.join(tracks_info).dropna()

            # by = 'log_nuc_OptoFGFR_intensity_mean_MEAN'
            by = 'log_receptor_normalized'
            field = 'mi_cross_per_slot'

            ax.set_xlim(-4, 6)
            ax.set_ylim(-1, 12)

            _, ax2 = plot_hist_with_rolling(DATA_MANAGER, tracks_mi_and_info, well_ids, field, by, ax=ax, field_label='bitrate', field_scale='bph', field_unit='bit/h', annotate=False)
            
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.xaxis.set_major_locator(MultipleLocator(1))
            # ax.xaxis.set_major_formatter(lambda x, pos: f"{x:+.0f}σ" if x != 0 else "μ")
            ax2.yaxis.set_visible(False)
            ax.set_title(set_to_label[set_id].replace('\nno inh', '').replace('+', '').replace('\n', ' + '), loc='left', fontweight='bold')
        
        for ax in axs[-1, :]:
            ax.set_xlabel('log optoFGFR fluorescence\nstandardized with mean and std')
        for ax in axs[:, 0]:
            ax.set_ylabel('bitrate [bit/h]')

        fig.savefig(str(output.svg))
        fig.savefig(str(output.png), dpi=300)

