import numpy as np
import pandas as pd

import matplotlib
from matplotlib import ticker
from subplots_from_axsize import subplots_from_axsize

from generic._defaults import (
    DATA_MANAGER,
    WELLS_SELECTED,
    SET_TYPES,
    DATASET_IDS,
    MODEL_IDS,
    TRAIN_IDS,
    TEST_IDS,
)

from src.per_track_plots import (
    plot_expanding_mean,
    plot_histogram,
    plot_rolling_mean,
    plot_sorted,
    get_fraction_transmitting,
    plot_hist_with_rolling,
)
from src.stat_utils import get_sorted
from src.fig_layout import row_to_pos, val_name_pos_list
from config.local_config import OUTPUT_PATH

matplotlib.use("agg")
workdir: OUTPUT_PATH

# CONGIGS

TEST_SET_TYPES = [
    'main',
    'main+cell+inh',
    'main+cell+inhtype',
]

TEST_IDS = ['q0', 'q1', 'q2']
TEST_IDS = ['q1', 'q1tr']


# INCLUDE CORE RULES

include: "generic/_core_rules.smk"
include: "generic/_join_results.smk"
include: "generic/_combine_plots.smk"


# AUXILIARY FUNCTIONS

def parse_wildcards(wildcards):
    per_well = wildcards.per == 'per_well'
    well_ids = [wildcards.well_or_set_id] if per_well else SET_ID_TO_WELL_IDS[wildcards.well_or_set_id] 
    experiments = [DATA_MANAGER.get_experiment(well_id) for well_id in well_ids]
    seconds_per_timepoint = DATA_MANAGER.get_seconds_per_timepoint_for_experiment_list(experiments)
    bph = 60 * 60 / seconds_per_timepoint / np.log(2)
    index_col = 'track_id' if per_well else ['well_id', 'track_id']
        
    return index_col, bph


# RULES

rule all:
    input:
        expand(
            'plot_per_track_analysis/combined/per_experiment/{experiment}/{plot_type}_{dataset_id}_{model_id}_{train_id}_{test_id}.png',
            plot_type=[
                # 'track-mi-vs-factors',
                # 'track-mi-vs-responsiveness',
                'factors-scatters',
                # 'track-mi-distribution',
                # # 'track-mi-distribution-vs-receptor',
                # # 'receptor-hist-cond-on-transmitting',
                # # 'responsiveness-hist-cond-on-transmitting',
                'rolling-mi-vs-receptor',
                'rolling-mi-vs-reporter',
                # # 'rolling-responsiveness-vs-receptor',
            ],
            experiment=WELLS_SELECTED['experiment'].unique(),
            dataset_id=DATASET_IDS,
            model_id=MODEL_IDS,
            train_id=TRAIN_IDS,
            test_id=TEST_IDS,
        ),
        expand(
            'plot_per_track_analysis/combined/per_set_type/{set_type}/{plot_type}_{dataset_id}_{model_id}_{train_id}_{test_id}.png',
            plot_type=[
                # 'track-mi-vs-factors',
                # 'track-mi-vs-responsiveness',
                'factors-scatters',
                # 'track-mi-distribution',
                # # 'track-mi-distribution-vs-receptor',
                # # 'receptor-hist-cond-on-transmitting',
                # # 'responsiveness-hist-cond-on-transmitting',
                'rolling-mi-vs-receptor',
                'rolling-mi-vs-reporter',
            ],
            set_type=TEST_SET_TYPES,
            dataset_id=DATASET_IDS,
            model_id=MODEL_IDS,
            train_id=TRAIN_IDS,
            test_id=TEST_IDS,
        ),
        expand(
            'plot_per_track_analysis/transmitting-fraction_{dataset_id}_{model_id}_{train_id}_{test_id}.png',
            dataset_id=DATASET_IDS,
            model_id=MODEL_IDS,
            train_id=TRAIN_IDS,
            test_id=TEST_IDS,
        )

rule plot_rolling_mi_vs_receptor:
    input:
        tracks_mi='cache/tracks_mi/{per}/{well_or_set_id}/tracks_mi_{dataset_id}_{model_id}_{train_id}_{test_id}.csv.gz',
        tracks_info='cache/preprocessed/{per}/{well_or_set_id}/tracks_info.csv.gz',
    output:
        'plot_per_track_analysis/single/{per}/{well_or_set_id}/rolling-mi-vs-receptor_{dataset_id}_{model_id}_{train_id}_{test_id}.png'
    run:
        per_well = wildcards.per == 'per_well'

        well_ids = [wildcards.well_or_set_id] if per_well else SET_ID_TO_WELL_IDS[wildcards.well_or_set_id]

        index_col = 'track_id' if per_well else ['well_id', 'track_id']
        tracks_mi = pd.read_csv(input.tracks_mi, index_col=index_col)
        tracks_info = pd.read_csv(input.tracks_info, index_col=index_col)
        tracks_mi_and_info = tracks_mi.join(tracks_info).dropna()

        # by = 'log_nuc_OptoFGFR_intensity_mean_MEAN'
        by = 'log_receptor_normalized'
        field = 'mi_cross_per_slot'

        fig, ax = subplots_from_axsize(top=.5)
        plot_hist_with_rolling(DATA_MANAGER, tracks_mi_and_info, well_ids, field, by, ax=ax, field_label='bitrate', field_scale='bph', field_unit='bit/h')
        
        fig.suptitle(f'{wildcards.well_or_set_id} -{wildcards.test_id}\n{wildcards.dataset_id}; {wildcards.model_id}; {wildcards.train_id};')
        fig.savefig(str(output), dpi=300)



rule plot_rolling_mi_vs_reporter:
    input:
        tracks_mi='cache/tracks_mi/{per}/{well_or_set_id}/tracks_mi_{dataset_id}_{model_id}_{train_id}_{test_id}.csv.gz',
        tracks_info='cache/preprocessed/{per}/{well_or_set_id}/tracks_info.csv.gz',
    output:
        'plot_per_track_analysis/single/{per}/{well_or_set_id}/rolling-mi-vs-reporter_{dataset_id}_{model_id}_{train_id}_{test_id}.png'
    run:
        per_well = wildcards.per == 'per_well'

        well_ids = [wildcards.well_or_set_id] if per_well else SET_ID_TO_WELL_IDS[wildcards.well_or_set_id]

        index_col = 'track_id' if per_well else ['well_id', 'track_id']
        tracks_mi = pd.read_csv(input.tracks_mi, index_col=index_col)
        tracks_info = pd.read_csv(input.tracks_info, index_col=index_col)
        tracks_mi_and_info = tracks_mi.join(tracks_info).dropna()

        by = 'log_reporter_normalized'
        field = 'mi_cross_per_slot'

        fig, ax = subplots_from_axsize(top=.5)
        plot_hist_with_rolling(DATA_MANAGER, tracks_mi_and_info, well_ids, field, by, ax=ax, field_label='bitrate', field_scale='bph', field_unit='bit/h', plot_thresholds=False)
        
        fig.suptitle(f'{wildcards.well_or_set_id} -{wildcards.test_id}\n{wildcards.dataset_id}; {wildcards.model_id}; {wildcards.train_id};')
        fig.savefig(str(output), dpi=300)


rule plot_rolling_responsiveness_vs_receptor:
    input:
        tracks_mi='cache/tracks_mi/{per}/{well_or_set_id}/tracks_mi_{dataset_id}_{model_id}_{train_id}_{test_id}.csv.gz',
        tracks_info='cache/preprocessed/{per}/{well_or_set_id}/tracks_info.csv.gz',
    output:
        'plot_per_track_analysis/single/{per}/{well_or_set_id}/rolling-responsiveness-vs-receptor_{dataset_id}_{model_id}_{train_id}_{test_id}.png'
    run:
        per_well = wildcards.per == 'per_well'

        well_ids = [wildcards.well_or_set_id] if per_well else SET_ID_TO_WELL_IDS[wildcards.well_or_set_id]

        index_col = 'track_id' if per_well else ['well_id', 'track_id']
        tracks_mi = pd.read_csv(input.tracks_mi, index_col=index_col)
        tracks_info = pd.read_csv(input.tracks_info, index_col=index_col)
        # tracks_info['log_receptor_MEAN_normalized'] = tracks_info['log_receptor_MEAN'] - tracks_info['log_receptor_MEAN'].mean()
        tracks_mi_and_info = tracks_mi.join(tracks_info).dropna()

        by = 'log_receptor_normalized'
        field = 'responsiveness'

        fig, ax = subplots_from_axsize(top=.5)
        plot_hist_with_rolling(DATA_MANAGER, tracks_mi_and_info, well_ids, field, by, ax=ax, field_label='mean response amplitude', field_scale=1., field_unit='')
        
        fig.suptitle(f'{wildcards.well_or_set_id} -{wildcards.test_id}\n{wildcards.dataset_id}; {wildcards.model_id}; {wildcards.train_id};')
        fig.savefig(str(output), dpi=300)


rule plot_track_mi_vs_factors:
    input:
        tracks_mi='cache/tracks_mi/{per}/{well_or_set_id}/tracks_mi_{dataset_id}_{model_id}_{train_id}_{test_id}.csv.gz',
        tracks_info='cache/preprocessed/{per}/{well_or_set_id}/tracks_info.csv.gz',
    output:
        'plot_per_track_analysis/single/{per}/{well_or_set_id}/track-mi-vs-factors_{dataset_id}_{model_id}_{train_id}_{test_id}.png'
    run:
        index_col, bph = parse_wildcards(wildcards)

        tracks_mi = pd.read_csv(input.tracks_mi, index_col=index_col)
        tracks_info = pd.read_csv(input.tracks_info, index_col=index_col)
        tracks_mi_and_info = tracks_mi.join(tracks_info)

        tracks_mi_and_info['track_start_h'] = tracks_mi_and_info['track_start'] / 60 / 60
        tracks_mi_and_info['track_end_h'] = tracks_mi_and_info['track_end'] / 60 / 60

        fig, axs = subplots_from_axsize(
            axsize=(3, 3),
            wspace=0.9,
            hspace=0.7,
            ncols=2,
            nrows=3,
            top=.5,
        )

        for ax, (col, xlabel) in zip(axs.ravel(), [
            ('track_start_h', 'Track start [hour]'),
            ('track_end_h', 'Track end [hour]'),
            ('X_responsiveness', 'mean reposnse amplitude'),
            ('X_ratio_std', 'X_ratio_std'),
            ('log_receptor_MEAN', 'mean log receptor intensity'),
            ('log_reporter_MEAN', 'mean log reporter intensity'),
        ]):
            ax.scatter(
                tracks_mi_and_info[col],
                tracks_mi_and_info['mi_cross_per_slot'] * bph,
                s=tracks_mi_and_info['slots'] / 30,
                color='k',
                edgecolor='none',
                alpha=0.1,
            )
            # if col == 'log_receptor_MEAN':
            #     ax.axvline(OPTOFGFR_THR, ls='--', color='k', alpha=.5)
            # if col == 'log_reporter_MEAN':
            #     ax.axvline(ERKKTR_THR, ls='--', color='k', alpha=.5)
            ax.set_xlabel(xlabel)
            

        for ax in axs[0, :]:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(1 / 6))

        for ax in axs.ravel():
            ax.set_ylabel('bitrate contribution [bit/h]')

        fig.suptitle(f'{wildcards.well_or_set_id} -{wildcards.test_id}\n{wildcards.dataset_id}; {wildcards.model_id}; {wildcards.train_id};')
        fig.savefig(str(output), dpi=300)



rule plot_track_mi_vs_responsiveness:
    input:
        tracks_mi='cache/tracks_mi/{per}/{well_or_set_id}/tracks_mi_{dataset_id}_{model_id}_{train_id}_{test_id}.csv.gz',
        tracks_info='cache/preprocessed/{per}/{well_or_set_id}/tracks_info.csv.gz',
    output:
        'plot_per_track_analysis/single/{per}/{well_or_set_id}/track-mi-vs-responsiveness_{dataset_id}_{model_id}_{train_id}_{test_id}.png'
    run:
        index_col, bph = parse_wildcards(wildcards)

        tracks_mi = pd.read_csv(input.tracks_mi, index_col=index_col)
        tracks_info = pd.read_csv(input.tracks_info, index_col=index_col)
        tracks_mi_and_info = tracks_mi.join(tracks_info)

        tracks_mi_and_info['track_start_h'] = tracks_mi_and_info['track_start'] / 60 / 60
        tracks_mi_and_info['track_end_h'] = tracks_mi_and_info['track_end'] / 60 / 60

        fig, ax = subplots_from_axsize(
            axsize=(3, 3),
            wspace=0.9,
            hspace=0.7,
            ncols=1,
            nrows=1,
            top=.5,
        )

        col = 'X_responsiveness'
        xlabel = 'mean response amplitude'

        ax.scatter(
            tracks_mi_and_info[col],
            tracks_mi_and_info['mi_cross_per_slot'] * bph,
            s=tracks_mi_and_info['slots'] / 30,
            color='k',
            edgecolor='none',
            alpha=0.1,
        )
        ax.set_xlabel(xlabel)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1 / 6))

        ax.set_ylabel('bitrate contribution [bit/h]')

        fig.suptitle(f'{wildcards.well_or_set_id} -{wildcards.test_id}\n{wildcards.dataset_id}; {wildcards.model_id}; {wildcards.train_id};')
        fig.savefig(str(output), dpi=300)


rule plot_factors_scatters:
    input:
        tracks_mi='cache/tracks_mi/{per}/{well_or_set_id}/tracks_mi_{dataset_id}_{model_id}_{train_id}_{test_id}.csv.gz',
        tracks_info='cache/preprocessed/{per}/{well_or_set_id}/tracks_info.csv.gz',
    output:
        'plot_per_track_analysis/single/{per}/{well_or_set_id}/factors-scatters_{dataset_id}_{model_id}_{train_id}_{test_id}.png'
    run:
        index_col, bph = parse_wildcards(wildcards)

        tracks_mi = pd.read_csv(input.tracks_mi, index_col=index_col)
        tracks_info = pd.read_csv(input.tracks_info, index_col=index_col)
        tracks_mi_and_info = tracks_mi.join(tracks_info)

        tracks_mi_and_info['track_start_h'] = tracks_mi_and_info['track_start'] / 60 / 60
        tracks_mi_and_info['track_end_h'] = tracks_mi_and_info['track_end'] / 60 / 60


        features = [
            # (('log_receptor_MEAN', 'mean log receptor intensity'), ('X_responsiveness', 'mean response amplitude'), ('mi_cross_per_slot', 'bitrate')),
            (('log_reporter_MEAN', 'mean log reporter intensity'), ('X_responsiveness', 'mean response amplitude'), ('mi_cross_per_slot', 'bitrate')),
            # (('log_receptor_MEAN', 'mean log receptor intensity'), ('log_reporter_MEAN', 'mean log reporter intensity'), ('X_responsiveness', 'mean response amplitude')),
            # (('log_receptor_MEAN', 'mean log receptor intensity'), ('log_reporter_MEAN', 'mean log reporter intensity'), ('mi_cross_per_slot', 'bitrate')),
        ]

        fig, axs = subplots_from_axsize(
            axsize=(3, 3),
            wspace=0.9,
            hspace=0.7,
            ncols=len(features),
            top=1.,
            right=1.,
            bottom=.8,
            squeeze=False,
        )

        for ax, ((xcol, xlabel), (ycol, ylabel), (ccol, clabel)) in zip(axs.ravel(), features):
            ax.scatter(
                tracks_mi_and_info[xcol],
                tracks_mi_and_info[ycol],
                c=tracks_mi_and_info[ccol],
                s=tracks_mi_and_info['slots'] / 30,
                edgecolor='none',
                alpha=0.1,
            )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f"color: {clabel}")


        fig.suptitle(f'{wildcards.well_or_set_id} -{wildcards.test_id}\n{wildcards.dataset_id}; {wildcards.model_id}; {wildcards.train_id};')
        fig.savefig(str(output), dpi=300)


rule plot_track_mi_distribution:
    input:
        tracks_mi='cache/tracks_mi/{per}/{well_or_set_id}/tracks_mi_{dataset_id}_{model_id}_{train_id}_{test_id}.csv.gz',
        tracks_info='cache/preprocessed/{per}/{well_or_set_id}/tracks_info.csv.gz',
    output:
        'plot_per_track_analysis/single/{per}/{well_or_set_id}/track-mi-distribution_{dataset_id}_{model_id}_{train_id}_{test_id}.png'
    run:
        index_col, bph = parse_wildcards(wildcards)

        tracks_mi = pd.read_csv(input.tracks_mi, index_col=index_col)
        tracks_info = pd.read_csv(input.tracks_info, index_col=index_col)

        fig, axs = subplots_from_axsize(
            axsize=(4, 3),
            wspace=1.0,
            ncols=3,
            top=.5,
        )

        field = 'mi_cross_per_slot'
        by = field

        field_sorted, slots_sorted = get_sorted(tracks_mi, field, by)

        field_cum = np.cumsum(field_sorted * slots_sorted) / slots_sorted.sum()
        slots_cum = np.cumsum(slots_sorted) / slots_sorted.sum()

        average, fraction_transmitting, average_transmitting, transmitting_thr = get_fraction_transmitting(field_sorted, slots_sorted)
        
        # MI v fraction of tracks
        plot_sorted(tracks_mi=tracks_mi, field=field, ax=axs[0], scale=bph)
        axs[0].axhline(average * bph, color='r', ls=':')
        # axs[0].axvline(fraction_transmitting, color='b', ls=':')
        axs[0].plot(fraction_transmitting, average * bph, marker='o', fillstyle='none', color='blue')

        # average MI v fractiqon of tracks
        plot_expanding_mean(tracks_mi=tracks_mi, field=field, ax=axs[1], scale=bph)
        axs[1].axhline(average * bph, color='r', ls=':')
        axs[1].axhline(average_transmitting * bph, 0, fraction_transmitting, color='g', ls=':')
        # axs[1].vlines([fraction_transmitting], 0, average_transmitting * bph, color='b', ls=':')
        axs[1].plot(fraction_transmitting, average_transmitting * bph, marker='o', fillstyle='none', color='blue')

        # MI dist
        plot_histogram(tracks_mi, field=field, field_label='MI [bit / hour]', ax=axs[2], scale=bph, show_transmitting=True)
        axs[2].axvline(average * bph, color='r', ls=':', label='average')
        axs[2].axvline(average_transmitting * bph, color='g', ls=':', label='average in transmitting')
        axs[2].axvline(transmitting_thr * bph, color='b', ls='-', lw=.5, label='transmitting threshold')
        axs[2].legend()
        

        fig.suptitle(f'{wildcards.well_or_set_id} -{wildcards.test_id}; {wildcards.dataset_id}; {wildcards.model_id}; {wildcards.train_id}')
        fig.savefig(str(output), dpi=300)


rule plot_track_mi_distribution_vs_receptor:
    input:
        tracks_mi='cache/tracks_mi/{per}/{well_or_set_id}/tracks_mi_{dataset_id}_{model_id}_{train_id}_{test_id}.csv.gz',
        tracks_info='cache/preprocessed/{per}/{well_or_set_id}/tracks_info.csv.gz',
    output:
        'plot_per_track_analysis/single/{per}/{well_or_set_id}/track-mi-distribution-vs-receptor_{dataset_id}_{model_id}_{train_id}_{test_id}.png'
    run:
        index_col, bph = parse_wildcards(wildcards)

        tracks_mi = pd.read_csv(input.tracks_mi, index_col=index_col)
        tracks_info = pd.read_csv(input.tracks_info, index_col=index_col)

        tracks_mi_and_info = tracks_mi.join(tracks_info)

        fig, axs = subplots_from_axsize(
            axsize=(4, 3),
            wspace=1.0,
            ncols=4,
            top=.5,
        )

        field = 'mi_cross_per_slot'
        by = 'log_receptor_MEAN'
        average = (tracks_mi_and_info[field] * tracks_mi_and_info['slots']).sum() / tracks_mi_and_info['slots'].sum()
        average_by = (tracks_mi_and_info[by] * tracks_mi_and_info['slots']).sum() / tracks_mi_and_info['slots'].sum()

        # cumulative MI v fraction of tracks
        plot_sorted(tracks_mi=tracks_mi_and_info, field=field, by=by, ax=axs[0], scale=bph)
        axs[0].axhline(average * bph, color='r', ls=':')
        
        # average MI v fraction of tracks
        plot_expanding_mean(tracks_mi=tracks_mi_and_info, field=field, by=by, ax=axs[1], scale=bph)
        axs[1].axhline(average * bph, color='r', ls=':')
        
        # MI v fraction of tracks
        plot_rolling_mean(tracks_mi=tracks_mi_and_info, field=field, by=by, ax=axs[2], scale=bph)
        axs[2].axhline(average * bph, color='r', ls=':')

        # MI dist
        plot_histogram(tracks_mi=tracks_mi_and_info, field=field, by=by, ax=axs[3], scale=1)
        axs[3].axvline(average_by, color='r', ls=':')
        
        fig.suptitle(f'{wildcards.well_or_set_id} -{wildcards.test_id}; {wildcards.dataset_id}; {wildcards.model_id}; {wildcards.train_id}')
        fig.savefig(str(output), dpi=300)


rule plot_receptor_hist_cond_on_transmitting:
    input:
        tracks_mi='cache/tracks_mi/{per}/{well_or_set_id}/tracks_mi_{dataset_id}_{model_id}_{train_id}_{test_id}.csv.gz',
        tracks_info='cache/preprocessed/{per}/{well_or_set_id}/tracks_info.csv.gz',
    output:
        'plot_per_track_analysis/single/{per}/{well_or_set_id}/receptor-hist-cond-on-transmitting_{dataset_id}_{model_id}_{train_id}_{test_id}.png'
    run:
        index_col, bph = parse_wildcards(wildcards)

        tracks_mi = pd.read_csv(input.tracks_mi, index_col=index_col)
        tracks_info = pd.read_csv(input.tracks_info, index_col=index_col)
        tracks_mi_and_info = tracks_mi.join(tracks_info)

        fig, ax = subplots_from_axsize(
            axsize=(4, 3),
            wspace=1.0,
            top=.5,
        )

        by = 'mi_cross_per_slot'
        field = 'log_receptor_MEAN'
         
        field_sorted, slots_sorted = get_sorted(tracks_mi_and_info, [by, field, 'slots'], by)

        slots_sorted_normalized = slots_sorted / slots_sorted.sum()
        field_cum = np.cumsum(field_sorted.mul(slots_sorted_normalized, axis=0))

        average = (tracks_mi_and_info[by] * tracks_mi_and_info['slots']).sum() / tracks_mi_and_info['slots'].sum()

        idx_max = np.searchsorted(-field_sorted[by], 0)
        idx_fraction_transmitting = np.searchsorted(field_cum[by].iloc[:idx_max], average)

        transmitting = field_sorted.iloc[:idx_fraction_transmitting]
        non_transmitting = field_sorted.iloc[idx_fraction_transmitting:]

        ax.hist(transmitting[field], bins=np.linspace(4.5, 8., 51), weights=transmitting['slots'], density=True, alpha=.4, color='olive', label='transmitting')
        ax.hist(non_transmitting[field], bins=np.linspace(4.5, 8., 51), weights=non_transmitting['slots'], density=True, alpha=.4, color='maroon', label='non-transmitting')
        
        ax.legend()

        fig.suptitle(f'{wildcards.well_or_set_id} -{wildcards.test_id};\n{wildcards.dataset_id}; {wildcards.model_id}; {wildcards.train_id}')
        fig.savefig(str(output), dpi=300)


rule plot_responsiveness_hist_cond_on_transmitting:
    input:
        tracks_mi='cache/tracks_mi/{per}/{well_or_set_id}/tracks_mi_{dataset_id}_{model_id}_{train_id}_{test_id}.csv.gz',
        tracks_info='cache/preprocessed/{per}/{well_or_set_id}/tracks_info.csv.gz',
    output:
        'plot_per_track_analysis/single/{per}/{well_or_set_id}/responsiveness-hist-cond-on-transmitting_{dataset_id}_{model_id}_{train_id}_{test_id}.png'
    run:
        index_col, bph = parse_wildcards(wildcards)
        
        tracks_mi = pd.read_csv(input.tracks_mi, index_col=index_col)
        tracks_info = pd.read_csv(input.tracks_info, index_col=index_col)
        tracks_mi_and_info = tracks_mi.join(tracks_info)

        fig, ax = subplots_from_axsize(
            axsize=(4, 3),
            wspace=1.0,
            top=.5,
        )

        by = 'mi_cross_per_slot'
        field = 'responsiveness'
        
        field_sorted, slots_sorted = get_sorted(tracks_mi_and_info, [by, field, 'slots'], by)

        slots_sorted_normalized = slots_sorted / slots_sorted.sum()
        field_cum = np.cumsum(field_sorted.mul(slots_sorted_normalized, axis=0))

        average = (tracks_mi_and_info[by] * tracks_mi_and_info['slots']).sum() / tracks_mi_and_info['slots'].sum()

        idx_max = np.searchsorted(-field_sorted[by], 0)
        idx_fraction_transmitting = np.searchsorted(field_cum[by].iloc[:idx_max], average)

        transmitting = field_sorted.iloc[:idx_fraction_transmitting]
        non_transmitting = field_sorted.iloc[idx_fraction_transmitting:]

        ax.hist(transmitting[field], bins=np.linspace(-.5, .5, 51), weights=transmitting['slots'], density=True, alpha=.4, color='olive', label='transmitting')
        ax.hist(non_transmitting[field], bins=np.linspace(-.5, .5, 51), weights=non_transmitting['slots'], density=True, alpha=.4, color='maroon', label='non-transmitting')
        
        ax.legend()

        fig.suptitle(f'{wildcards.well_or_set_id} -{wildcards.test_id};\n{wildcards.dataset_id}; {wildcards.model_id}; {wildcards.train_id}')
        fig.savefig(str(output), dpi=300)


rule plot_transmitting_analysis:
    input:
        transmitting_analysis_all='plot_per_track_analysis/transmitting_analysis_all.csv'
    output:
        fraction_transmitting='plot_per_track_analysis/transmitting-fraction_{dataset_id}_{model_id}_{train_id}_{test_id}.png',
        mi_ce_transmitting='plot_per_track_analysis/transmitting-mi_{dataset_id}_{model_id}_{train_id}_{test_id}.png',
    run:
        transmitting_analysis = pd.read_csv(input.transmitting_analysis_all)

        train_ids = wildcards.train_id.split('+')
        max_train_ids = len([col for col in transmitting_analysis.columns if col.startswith('train_id_')])

        transmitting_analysis_plot = transmitting_analysis[
            (transmitting_analysis['dataset_id'] == wildcards.dataset_id)
          & (transmitting_analysis['model_id'] == wildcards.model_id)
          & (transmitting_analysis[[f'train_id_{it}' for it in range(len(train_ids))]] == train_ids).all(axis=1)
          & (transmitting_analysis[[f'train_id_{it}' for it in range(len(train_ids), max_train_ids)]].isna()).all(axis=1)
          & (transmitting_analysis['test_id'] == wildcards.test_id)
        ].copy()

        title = '; '.join([
            wildcards.dataset_id,
            wildcards.model_id,
            wildcards.train_id,
            wildcards.test_id,
        ])

        transmitting_analysis_plot['pos'] = transmitting_analysis_plot.apply(row_to_pos, axis=1)

        for field, label, ymin, ymax in [
            ('fraction_transmitting', 'Fraction of transmitting cells', 0., 1.), 
            ('mi_ce_transmitting', 'Bitrate [bit / h]', -.5, 11.),
            ]:
            fig, ax = subplots_from_axsize(axsize=(8, 3), top=0.4)

            for experiment, transmitting_analysis_plot_exp in transmitting_analysis_plot.groupby('experiment'):
                ax.plot(
                    transmitting_analysis_plot_exp['pos'],
                    transmitting_analysis_plot_exp[field],
                    'o',
                    alpha=0.5,
                    label=experiment,
                )

            ax.legend()
            ax.set_xlim(0.5, 13.0)
            bottom = min(ymin, 1.05* transmitting_analysis_plot[field].replace(-np.inf, np.nan).min())
            top = max(ymax, 1.05* transmitting_analysis_plot[field].replace(np.inf, np.nan).max())
            ax.set_ylim(bottom, top)
            ax.grid(color='k', alpha=0.5, ls=':')
            ax.set_xticks(*list(zip(*[(pos, name) for _, name, pos in val_name_pos_list])))
            ax.set_ylabel(label)
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_title(title)

            fig.savefig(str(output[field]))

