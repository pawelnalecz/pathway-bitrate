import matplotlib
from subplots_from_axsize import subplots_from_axsize
import pandas as pd

from generic._defaults import (
    DATA_MANAGER,
    WELLS_SELECTED,
    DATASET_IDS,
)

from config import parameters
from config.local_config import OUTPUT_PATH

matplotlib.use("agg")
workdir: OUTPUT_PATH


SET_TYPES = []


# INCLUDE CORE RULES

include: "generic/_combine_plots.smk"
include: "generic/_core_rules.smk"

# RULES

rule all:
    input:
        expand(
            [
                # 'quality_check/combined/per_experiment/{experiment}/scatter_receptor_erkktr.png',
                'quality_check/combined/per_experiment/{experiment}/receptor_hist_absolute.png',
                'quality_check/combined/per_experiment/{experiment}/reporter_hist_absolute.png',
                'quality_check/combined/per_experiment/{experiment}/receptor_hist_normalized.png',
            ],
            experiment=WELLS_SELECTED['experiment'].unique(),
        ),
        expand(
             [
                'quality_check/combined/per_set_type/{set_type}/receptor_hist_absolute.png',
                'quality_check/combined/per_set_type/{set_type}/receptor_hist_normalized.png',
            ],
            set_type=SET_TYPES,
        ),
        'quality_check/experiment_summary_all.csv',
        'quality_check/experiment_summary_all.html',



rule plot_scatter_receptor_erkktr:
    input:
        tracks_info='cache/preprocessed/{per}/{well_or_set_id}/tracks_info.csv.gz',
    output:
        'quality_check/single/{per}/{well_or_set_id}/scatter_receptor_erkktr.png',
    resources:
        mem_mib=512
    run:
        tracks_info = pd.read_csv(input.tracks_info)

        fig, ax = subplots_from_axsize(
            axsize=(3, 3),
            left=1.2,
            right=1.2,
            top=0.5,
        )

        ax.scatter(
            tracks_info['log_receptor_MEAN'],
            tracks_info['log_reporter_MEAN'],
            s=tracks_info['track_length'] / 30,
            color='k',
            edgecolor='none',
            alpha=0.1,
        )
        # ax.axvline(parameters.OPTOFGFR_THR, color='r', ls=':')
        # ax.axhline(parameters.ERKKTR_THR, color='r', ls=':')
        ax.set_xlabel('log receptor')
        ax.set_ylabel('log reporter')

        ax.set_title(f"{wildcards.well_or_set_id}")

        fig.savefig(str(output), dpi=300)



rule plot_receptor_hist:
    input:
        tracks_info='cache/preprocessed/{per}/{well_or_set_id}/tracks_info.csv.gz',
    output:
        'quality_check/single/{per}/{well_or_set_id}/receptor_hist_{normalized}.png',
    resources:
        mem_mib=512
    run:
        tracks_info = pd.read_csv(input.tracks_info)

        fig, ax = subplots_from_axsize(
            axsize=(3, 3),
            left=1.2,
            right=1.2,
            top=0.5,
        )

        normalized = wildcards.normalized == 'normalized'
        ax.hist(
            tracks_info['log_receptor_normalized' if normalized else 'log_receptor_MEAN'], 
            weights=tracks_info['track_length'],
            density=True,
            color='k',
            bins=51,
        )
        ax.set_title(f"{wildcards.well_or_set_id}")

        fig.savefig(str(output), dpi=300)


rule plot_reporter_hist:
    input:
        tracks_info='cache/preprocessed/{per}/{well_or_set_id}/tracks_info.csv.gz',
    output:
        'quality_check/single/{per}/{well_or_set_id}/reporter_hist_{normalized}.png',
    resources:
        mem_mib=512
    run:
        tracks_info = pd.read_csv(input.tracks_info)

        fig, ax = subplots_from_axsize(
            axsize=(3, 3),
            left=1.2,
            right=1.2,
            top=0.5,
        )

        normalized = wildcards.normalized == 'normalized'
        ax.hist(
            tracks_info['log_reporter_normalized' if normalized else 'log_reporter_MEAN'], 
            weights=tracks_info['track_length'],
            density=True,
            color='k',
            bins=51,
        )
        ax.set_title(f"{wildcards.well_or_set_id}")

        fig.savefig(str(output), dpi=300)



rule experiment_summary_all:
    input:
        expand(
            'cache/summary/{well_id}/experiment_summary.csv',
            well_id=WELLS_SELECTED.index.get_level_values('well_id'),
        )
    output:
        'quality_check/experiment_summary_all.csv',
        'quality_check/experiment_summary_all.html',
    run:
        results = pd.concat([
            pd.read_csv(str(path)) for path in input
        ])
        results.to_csv(str(output[0]), index=False)
        results.to_html(str(output[1]), index=False, 
                        formatters={
                            col: ("{:.1f}".format if '(tpt)' in col else "{:.4g}".format) 
                            for col in results.select_dtypes('float')
                        }
                    )


rule experiment_summary:
    input:
        tracks_info='cache/preprocessed/per_well/{well_id}/tracks_info.csv.gz',
    output:
        'cache/summary/{well_id}/experiment_summary.csv',
    resources:
        mem_gib=2
    run:
        well_id = wildcards.well_id
        experiment = DATA_MANAGER.get_experiment(well_id)
        well_info = DATA_MANAGER.get_well_info(well_id)
        experiment_info = DATA_MANAGER.get_experiment_info(experiment)
        seconds_per_timepoint = DATA_MANAGER.get_seconds_per_timepoint(experiment)

        receptor_channel = experiment_info['receptor_channel']

        tracks_info = pd.read_csv(input['tracks_info']).set_index('track_id')

        experiment_duration_in_tp = int((tracks_info['track_end'].max() - tracks_info['track_start'].min()) / seconds_per_timepoint) + 1
        quality_data_points_per_frame = tracks_info.groupby('quality')['track_length'].sum().reindex([-1, 0,1,2], fill_value=0) / experiment_duration_in_tp
        tracks_info['is_low_receptor']  = tracks_info['log_receptor_normalized'] < experiment_info[f'{receptor_channel}_lower_thr']
        tracks_info['is_high_receptor'] = tracks_info['log_receptor_normalized'] > experiment_info[f'{receptor_channel}_upper_thr']
        tracks_info['is_short_track'] = tracks_info['track_length'] < parameters.TRACK_LENGTH_THR
        
        experiment_summary = {
            'well_id': well_id,
            **well_info, 
            'n tracks': len(tracks_info),
            # 'experiment_length [s]': experiment_duration_in_tp * seconds_per_timepoint,
            'experiment length [min]': experiment_duration_in_tp * seconds_per_timepoint / 60,
            # 'experiment_length [h]': experiment_duration_in_tp * seconds_per_timepoint / 3600,
            'all tracks (tpt)': tracks_info['track_length'].sum() / experiment_duration_in_tp,
            'quality >=0 (tpt)': quality_data_points_per_frame[quality_data_points_per_frame.index >= 0].sum(),
            'quality >=1 (tpt)': quality_data_points_per_frame[quality_data_points_per_frame.index >= 1].sum(),
            'quality >=2 (tpt)': quality_data_points_per_frame[quality_data_points_per_frame.index >= 2].sum(),
            '(low receptor (tpt))': tracks_info.groupby('is_low_receptor')['track_length'].sum().reindex([False, True], fill_value=0)[True] / experiment_duration_in_tp,
            '(high receptor (tpt))': tracks_info.groupby('is_high_receptor')['track_length'].sum().reindex([False, True], fill_value=0)[True] / experiment_duration_in_tp,
            # '(low reporter (tpt))': tracks_info.groupby('is_low_reporter')['track_length'].sum().reindex([False, True], fill_value=0)[True] / experiment_duration_in_tp,
            # '(overexposed (tpt))': tracks_info[tracks_info['is_overexposed']]['track_length'].sum() / experiment_duration_in_tp,
            '(receptor below background (tpt))': tracks_info[tracks_info['is_receptor_below_background']]['track_length'].sum() / experiment_duration_in_tp,
            '(reporter below background (tpt))': tracks_info[tracks_info['is_reporter_below_background']]['track_length'].sum() / experiment_duration_in_tp,
            'receptor background': tracks_info['receptor_background_MEAN'].mean(),
            'reporter background': tracks_info['reporter_background_MEAN'].mean(),
        }
        experiment_summary_df = pd.DataFrame([experiment_summary], index=[0]).set_index('well_id')
        experiment_summary_df.to_csv(str(output))


