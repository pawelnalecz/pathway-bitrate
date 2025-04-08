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
from src.fig_layout import row_to_pos, val_name_pos_list

matplotlib.use("agg")
workdir: OUTPUT_PATH


# CONFIGS

set_id = 'main'

# INCLUDE CORE RULES

include: "generic/_core_rules.smk"

# RULES

rule fig_2A:
    input:
        inputs_set([
            'cache/preprocessed/per_well/{well_id}/tracks_preprocessed.pkl.gz'
        ], 'main')
    output:
        svg='figures/panels/fig2A.svg',
        png='figures/panels/fig2A.png',
    resources:
        mem_gib=lambda wc, input: 2 * len(input),
    run:
        wells = list(SET_ID_TO_WELL_IDS[set_id])
        inputs = pd.Series(list(input), index=wells)

        experiments = np.unique([DATA_MANAGER.get_experiment(well_id) for well_id in wells])
        pulses = DATA_MANAGER.get_pulses(experiments[0])['time_in_seconds']

        for experiment in experiments:
            assert DATA_MANAGER.get_pulses(experiment)['time_in_seconds'].equals(pulses), f"Experiment {experiment} has incompatible pulses"

        fig, ax = subplots_from_axsize(axsize=(2.6, 6), left=.8, top=0.4, right=.02, bottom=.8)

        offset_per_experiment = 1.2
        xmin = 15000
        xmax = 24000

        inhibotor_cols = DATA_MANAGER.inhibitor_cols
        it = 0
        for well_group_id, well_group in DATA_MANAGER.wells.loc[wells].groupby(['cell_line'] + inhibotor_cols):
            well_group_id_series = pd.Series(well_group_id, index=['cell_line'] + inhibotor_cols)
            offset = - offset_per_experiment * row_to_pos(well_group_id_series)
            color = 'goldenrod' if well_group_id_series['cell_line'] == 'BEAS2B' else 'maroon'
            trks_preprocessed = pd.concat([
                    pd.read_pickle(inputs[wid])
                    for wid in well_group.index
                ],
                names=['well_id'],
                keys=well_group.index,
            )

            trks_preprocessed = trks_preprocessed[
                (trks_preprocessed.index.get_level_values('time_in_seconds') >= xmin)
            & (trks_preprocessed.index.get_level_values('time_in_seconds') <= xmax)
                ]
            trks_preprocessed['log_translocation'] = np.log(trks_preprocessed['translocation'])
            trks_preprocessed['log_translocation_normalized'] = trks_preprocessed['log_translocation'] - trks_preprocessed['log_translocation'].mean()
            
            for wid, row in well_group.iterrows():
                for track_id in trks_preprocessed.loc[wid].index.unique(level='track_id')[:10]:
                    trk = trks_preprocessed.loc[(wid, track_id)]
                    (trk - trk.mean() + offset).reset_index().plot('time_in_seconds', 'log_translocation_normalized', ax=ax, label='', color='k', alpha=.05)
            
            (trks_preprocessed.groupby('time_in_seconds')['log_translocation_normalized'].mean() + offset).plot(
                ax=ax, lw=2, label=wid, 
                color=color,
                # color=plt.get_cmap('rainbow')( 0.02 * it),
            )

            ax.vlines(pulses[pulses.between(xmin, xmax)], offset -.3, offset + .3, color='k', ls='--', alpha=.3)

            it += 1
            
        ax.set_xlim(xmin, xmax)
        ax.set_yticks(*list(zip(*[(-offset_per_experiment * pos, name) for _, name, pos in val_name_pos_list])))
        ax.xaxis.set_major_locator(MultipleLocator(1800))
        ax.xaxis.set_minor_locator(MultipleLocator(600))
        ax.xaxis.set_major_formatter(lambda x, pos: f"{x // 3600:.0f}" if not x % 3600 else '')
        ax.get_legend().set_visible(False)
        ax.spines[['top', 'left', 'right']].set_visible(False)
        ax.set_xlabel('time [h]')
        maxpos = max([pos for _, _, pos in val_name_pos_list])
        minpos = min([pos for _, _, pos in val_name_pos_list])
        ax.set_ylim(-offset_per_experiment * (maxpos + .5), -offset_per_experiment * (minpos - .5))

        fig.savefig(str(output.svg))
        fig.savefig(str(output.png), dpi=300)

