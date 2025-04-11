import numpy as np

import matplotlib
from matplotlib import ticker
from subplots_from_axsize import subplots_from_axsize

from generic._defaults import (
    DATA_MANAGER,
    WELLS,
    SET_TYPES,
)

from src.data_preprocessing import tracks_transform_column
from src.internal_abbreviations import format_positive_inhibitors

from config.local_config import OUTPUT_PATH

matplotlib.use("agg")
workdir: OUTPUT_PATH


# CONFIGS

WELLS_SELECTED = WELLS[
    WELLS['experiment'].isin([
        '2023-08-28-BEAS2B--intensities',
        '2024-01-08-STE1',
        '2024-07-30-STE1',
        '2024-08-08-STE1',
        # '2024-09-08-BEAS2B',
        # '2024-09-18-BEAS2B-bad',

        '2024-10-22-BEAS2B',
        # '2024-11-20-BEAS2B',
        '2024-11-20-BEAS2B--first-11h',
        '2024-11-27-BEAS2B',
        '2024-12-23-BEAS2B',
    ])
]

USE_ALL_TRACKS_FOR_WELLS = WELLS[
    WELLS['experiment'].isin([
        '2023-08-28-BEAS2B--intensities',
    ])].index.tolist()


FIELDS = {
    #'translocation': dict(col='translocation'),
    #'win5_translocation': dict(col='translocation', rolling_min_window=5),
    'log_translocation': dict(col='translocation', log_transform=True),
    #'log_win5_translocation': dict(col='translocation', log_transform=True, rolling_min_window=5),
    #'d_log_translocation': dict(col='translocation', differentiate=True, log_transform=True),
    #'d_log_win5_translocation': dict(col='translocation', differentiate=True, log_transform=True, rolling_min_window=5),
}

# INCLUDE CORE RULES

include: "generic/_combine_plots.smk"
include: "generic/_core_rules.smk"


# RULES

rule all:
    input:
        expand(
            'plot_tracks_previews/combined/per_experiment/{experiment}/preview_{field}.png',
            field=FIELDS,
            experiment=WELLS_SELECTED['experiment'].unique(),
        )


rule plot_tracks_preview:
    input:
        tracks_preprocessed=lambda wildcards: f"cache/preprocessed/per_well/{wildcards.well_id}/tracks_preprocessed{'_all' if wildcards.well_id in USE_ALL_TRACKS_FOR_WELLS else ''}.pkl.gz",
    output:
        'plot_tracks_previews/single/per_well/{well_id}/preview_{field}.png'
    resources:
        mem_gib=1
    run:
        well_id = wildcards.well_id
        field = wildcards.field
        well_info = DATA_MANAGER.get_well_info(well_id)
        experiment = well_info['experiment']
        seconds_per_timepoint = DATA_MANAGER.get_seconds_per_timepoint(experiment)

        tracks = pd.read_pickle(input.tracks_preprocessed)
        pulses = DATA_MANAGER.get_pulses(experiment)
        valid_end = DATA_MANAGER.experiments.loc[experiment]['valid_end']

        tracks['time_in_hours'] = tracks.index.get_level_values('time_in_seconds') / 60 / 60
        time_end = tracks['time_in_hours'].max()

        tracks[field] = tracks_transform_column(
            tracks,
            **FIELDS[field],
            seconds_per_timepoint=seconds_per_timepoint,
        )

        fig, ax = subplots_from_axsize(
            axsize=(time_end / 3, 2),
            top=0.5,
        )

        # plot individual tracks
        tracks.groupby('track_id').plot(
            'time_in_hours',
            field,
            ax=ax,
            alpha=0.05,
            color='k',
            lw=0.5,
            legend=False,
        )

        # plot mean
        tracks.groupby('time_in_hours')[field].mean().plot(
            ax=ax,
            alpha=0.5,
            color='r',
            legend=False,
        )

        # plot pulses
        for _, pulse in pulses.iterrows():
            ax.axvline(
                pulse.time_in_seconds / 60 / 60,
                color='b' if pulse.valid else 'r',
                ls='--',
                lw=0.5,
                alpha=0.5,
            )

        inhibitor_str = format_positive_inhibitors(well_info, unit_separator=' ',  inhibitors_separator=' + ', no_inh='no inhibitor')

        ax.set_title(
            f'{well_info["experiment"]}; '
            f'exposition {well_info["exposition"]};\n'
            f'{inhibitor_str}; '
            f'rep {well_info["repetition"]}'
        )
        ax.set_xlabel('time [hour]')
        ax.set_ylabel(field)

        bottom = min(tracks[field].pipe(lambda x: x[x > -np.inf]).min(), 0)
        top = tracks[field].max()

        ax.set_xlim(0, time_end)
        ax.set_ylim(bottom, top)

        # mark unused in gray (where L or I is not defined)
        l_or_i_nan = (tracks['I'].isna() | tracks['L'].isna()).groupby('time_in_seconds').all()
        if l_or_i_nan.any():
            ax.fill_between(
                l_or_i_nan.index / 60 / 60,
                facecolor='grey',
                y1=bottom,
                y2=np.where(l_or_i_nan, top, bottom),
                edgecolor='none',
                alpha=0.2,
            )

        # mark invalid in red
        all_invalid = (~tracks['valid']).groupby('time_in_seconds').all()
        if all_invalid.any():
            ax.fill_between(
                all_invalid.index / 60 / 60,
                facecolor='r',
                y1=bottom,
                y2=np.where(all_invalid, top, bottom),
                edgecolor='none',
                alpha=0.2,
            )

        ax.spines[['top', 'right']].set_visible(False)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1 / 6))

        fig.savefig(str(output), dpi=600)

