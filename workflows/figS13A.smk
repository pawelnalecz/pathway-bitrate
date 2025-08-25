import matplotlib
from subplots_from_axsize import subplots_from_axsize
import pandas as pd
import json
import re


from generic._defaults import (
    DATA_MANAGER,
    WELLS_SELECTED,
    SET_TYPES,
    DATASET_IDS,
    MODEL_IDS,
    TRAIN_IDS,
    TEST_IDS,
)

# from config.configs import (
#     DATASET_CONFIGS,
# )


from src.fig_layout import set_to_label
from config.local_config import OUTPUT_PATH

matplotlib.use("agg")
workdir: OUTPUT_PATH


# INCLUDE CORE RULES

include: "generic/_combine_plots.smk"
include: "generic/_core_rules.smk"


# CONFIGS


slice_start = 1
SLICE_END_RANGE = range(1, 13)
MAX_SLICE_LENGTH = 12

SLICES = {
    f'r{slice_start}-r{slice_end}': {
         'r_ts': list(range(60*slice_start, 60*(slice_end + 1), 60)),
    }
    for slice_end in SLICE_END_RANGE
    if 0 < slice_end - slice_start and slice_end - slice_start < MAX_SLICE_LENGTH
}

DATASET_CONFIGS = {
    f'{dataset_id}-{slice_id}': DATASET_CONFIGS[dataset_id] | slice_config
    for slice_id, slice_config in SLICES.items() 
    for dataset_id in DATASET_IDS
}

DATASET_IDS = [
    f'{dataset_id}-{slice_id}'
    for slice_id in SLICES.keys() 
    for dataset_id in DATASET_IDS
]


set_types_and_colors = [
    ('main+STE1+0uM',        'slategray'),
    ('main+STE1+criz03uM',   'deepskyblue'),
    ('main+BEAS2B+0uM',      'goldenrod'),
    ('main+BEAS2B+cycl1uM',     'gold'),
    ('main+BEAS2B+tram05uM',     'red'),
    # ('main+BEAS2B+criz03uM', 'sandybrown'),
    # ('main+BEAS2B+tram05uMcycl1uM', 'navajowhite'),
]


# INCLUDE JOINING (must go after DATASET_CONFIGS are redefined)

include: "generic/_join_results.smk"

# RULES

rule fig_S13A:
    input:
        'figS13A/mi_all.csv'
    output:
        svg='figures/panels/figS13A.svg',
        png='figures/panels/figS13A.png',
    run:
        mi_all = pd.read_csv(str(input))

        expr = re.compile(r"r\-?[0-9]+\-r\-?[0-9]+\-(.+)")

        mi_all['slice_start'] = mi_all['r_ts'].apply(lambda x: json.loads(x)[0]  / 60)
        mi_all['slice_end']   = mi_all['r_ts'].apply(lambda x: json.loads(x)[-1] / 60)


        fig, axs = subplots_from_axsize(ncols=len(set_types_and_colors), axsize=(2,2), sharex=True, sharey=True, bottom=.6, top=.6, left=1., right=1.)

        for ax, (set_id, color) in zip(axs, set_types_and_colors):
            mi_set = mi_all[mi_all['well_id'].isin(SET_ID_TO_WELL_IDS[set_id])]
            mi_set.set_index(['slice_end', 'well_id'])['mi_ce'].unstack('well_id').plot(lw=1, alpha=.3, color=color, ax=ax)
            mi_set.groupby('slice_end')['mi_ce'].mean().plot(lw=2, color=color, ax=ax)
            ax.get_legend().set_visible(False)
            ax.set_title(set_to_label[set_id], fontsize='medium') #.replace('\n', ' + ')
            ax.set_xlabel('')

            
        fig.supxlabel('slice end [min after pusle]')
        fig.supylabel('bitrate [bit/h]')


        fig.savefig(str(output.svg))
        fig.savefig(str(output.png), dpi=300)

