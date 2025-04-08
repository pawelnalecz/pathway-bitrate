import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.data_manager import DataManager
from src.stat_utils import get_sorted


def get_fraction_transmitting(field_sorted, slots_sorted):
    field_cum = np.cumsum(field_sorted * slots_sorted) / slots_sorted.sum()
    slots_cum = np.cumsum(slots_sorted) / slots_sorted.sum()
    average = (field_sorted * slots_sorted).sum() / slots_sorted.sum()
    idx_max = np.searchsorted(-field_sorted, 0)
    idx_fraction_transmitting = np.searchsorted(field_cum.iloc[:idx_max], average)

    fraction_transmitting = slots_cum.iloc[idx_fraction_transmitting]
    average_transmitting = average / fraction_transmitting
    transmitting_thr = field_sorted.iloc[idx_fraction_transmitting]

    return average, fraction_transmitting, average_transmitting, transmitting_thr


def get_rolling_average(sorted_df, field, by, weights, weights_thr):

    i1 = 0
    i2 = 0
    n = len(sorted_df)
    field_sum = 0
    by_sum = 0
    weights_sum = 0

    field_iter1 = iter(sorted_df[field].astype('float64'))
    by_iter1 = iter(sorted_df[by].astype('float64'))
    weights_iter1 = iter(sorted_df[weights])

    field_iter2 = iter(sorted_df[field].astype('float64'))
    by_iter2 = iter(sorted_df[by].astype('float64'))
    weights_iter2 = iter(sorted_df[weights])

    rolling_avg_parts = []
    while i2 < n:
        if weights_sum < weights_thr:
            next_weight = next(weights_iter2)
            field_sum += next_weight * next(field_iter2)
            by_sum    += next_weight * next(by_iter2)
            weights_sum += next_weight
            i2 += 1
        else:
            next_weight = next(weights_iter1)
            field_sum -= next_weight * next(field_iter1)
            by_sum    -= next_weight * next(by_iter1)
            weights_sum -= next_weight
            i1 += 1
        rolling_avg_parts.append((field_sum, by_sum, weights_sum))
    while i1 < n:
        next_weight = next(weights_iter1)
        field_sum -= next_weight * next(field_iter1)
        by_sum    -= next_weight * next(by_iter1)
        weights_sum -= next_weight
        i1 += 1
        rolling_avg_parts.append((field_sum, by_sum, weights_sum))

    rolling_avg = pd.DataFrame(rolling_avg_parts, columns=[field + '_sum', by + '_sum', weights])
    rolling_avg[field] = rolling_avg[field + '_sum'] / rolling_avg[weights]
    rolling_avg[by] = rolling_avg[by + '_sum'] / rolling_avg[weights]
    del rolling_avg[field + '_sum']
    del rolling_avg[by + '_sum']
    return rolling_avg


def plot_sorted(tracks_mi, field, by=None, ax=None, scale=1):
    if by is None:
        by = field

    if ax is None:
        ax = plt.gca()

    field_sorted, slots_sorted = get_sorted(tracks_mi, field, by)

    field_cum = np.cumsum(field_sorted * slots_sorted) / slots_sorted.sum()
    slots_cum = np.cumsum(slots_sorted) / slots_sorted.sum()

    ax.plot(slots_cum, field_cum * scale, 'k')
    
    ax.set_xlabel('fraction of tracks included (weighted by slots)')
    ax.set_ylabel('cumulative MI [bit / hour]')


def plot_expanding_mean(tracks_mi, field, by=None, ax=None, scale=1):
    if by is None:
        by = field

    if ax is None:
        ax = plt.gca()

    field_sorted, slots_sorted = get_sorted(tracks_mi, field, by)

    field_expanding_mean = np.cumsum(field_sorted * slots_sorted) / np.cumsum(slots_sorted)
    slots_cum = np.cumsum(slots_sorted) / slots_sorted.sum()

    ax.plot(slots_cum, field_expanding_mean * scale, 'k')

    ax.set_xlabel('fraction of tracks included (weighted by slots)')
    ax.set_ylabel('average MI [bit / hour]')


def plot_rolling_mean(tracks_mi, field, by=None, win=100, ax=None, scale=1):
    if by is None:
        by = field

    if ax is None:
        ax = plt.gca()

    field_sorted, slots_sorted = get_sorted(tracks_mi, field, by)

    field_rolling_mean = (field_sorted * slots_sorted).rolling(win, center=True).sum() / slots_sorted.rolling(win, center=True).sum()
    slots_cum = np.cumsum(slots_sorted) / slots_sorted.sum()

    ax.plot(slots_cum, field_rolling_mean * scale, 'k')

    ax.set_xlabel('fraction of tracks included (weighted by slots)')
    ax.set_ylabel('average MI [bit / hour]')


def plot_histogram(tracks_mi, field, by=None, field_label=None, ax=None, scale=1, show_transmitting=False):
    if by is None:
        by = field

    if ax is None:
        ax = plt.gca()
        
    if field_label is None:
        field_label = by

    ax.hist(
        tracks_mi[by] * scale,
        weights=tracks_mi['slots'],
        density=True,
        color='k',
        bins=51,
    )

    ax.set_xlabel(field_label)
    ax.set_ylabel('p.d.f. (weighted by slots)')


def plot_hist_with_rolling(data_manager: DataManager, tracks_mi, well_ids, field, by, ax, win_tpt=50, field_label=None, field_scale='bph', field_unit='bit/h', plot_thresholds=True, annotate=True):
    field_label = field_label or field

    experiments = [data_manager.get_experiment(well_id) for well_id in  well_ids]
    seconds_per_timepoint = data_manager.get_seconds_per_timepoint_for_experiment_list(experiments)
    starts_ends = [data_manager.get_effective_time_range(experiment) for experiment in experiments]
    starts, ends = list(zip(*starts_ends))
    start = min(starts)
    end = max(ends)
    receptor_thrs = pd.DataFrame([data_manager.get_receptor_thresholds(experiment) for experiment in experiments], index=well_ids)

    bph = 60 * 60 / seconds_per_timepoint / np.log(2)
    field_scale = bph if field_scale == 'bph' else field_scale
    experiment_length = (end - start) // seconds_per_timepoint

    slotssum_thr = win_tpt * experiment_length
    rolling_avg = get_rolling_average(tracks_mi.sort_values(by), field, by, 'slots', slotssum_thr)

    ax.plot(rolling_avg[by], rolling_avg[field] * field_scale)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    if plot_thresholds:
        for rlt in receptor_thrs['receptor_lower_thr'].unique():
            ax.axvline(rlt, color='olive', ls='--', alpha=.3)
        for rut in receptor_thrs['receptor_upper_thr'].unique():
            ax.axvline(rut, color='olive', ls='--', alpha=.3)
        ax.fill_betweenx(ylim, receptor_thrs['receptor_lower_thr'].max(), receptor_thrs['receptor_upper_thr'].min(), color='olive', alpha=.2)

    ax2 = ax.twinx()
    ax2.hist(tracks_mi[by], weights=tracks_mi['slots'], bins=np.linspace(*xlim, 51), density=True, alpha=.3, color='k')

    ax.set_ylim(ylim)
    ax.set_xlabel(by)
    ax.set_ylabel(field)

    if len(well_ids) == 1:
        is_between_thresholds = tracks_mi[by].between(receptor_thrs['receptor_lower_thr'].max(), receptor_thrs['receptor_upper_thr'].min())
    else:
        is_between_thresholds = (
            tracks_mi
                .join(receptor_thrs, on='well_id')
                .pipe(lambda x: x[by].between(x['receptor_lower_thr'], x['receptor_upper_thr']))
        )
    tracks_in_range = tracks_mi[is_between_thresholds]
        
    tpt_all = tracks_mi['slots'].sum() / experiment_length
    tpt_in_range = tracks_in_range['slots'].sum() / experiment_length

    field_avg_in_range = (tracks_in_range['slots'] * tracks_in_range[field]).sum() / tracks_in_range['slots'].sum()

    if plot_thresholds and annotate:
        ax.annotate(
            f"tpt in range: {tpt_in_range:.1f} ({tpt_in_range / tpt_all:.2%})\n"
            f"{field_label} in range: {field_avg_in_range * field_scale:.2f} {field_unit}",
            (0.04, 0.8), xycoords='axes fraction', horizontalalignment='left', verticalalignment='center')
    return ax, ax2


        

