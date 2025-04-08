import random
import warnings

import numpy as np
import pandas as pd
from numpy.random import MT19937, RandomState, SeedSequence

from scipy.special import expit

import jax.numpy as jnp

from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Rectangle, Polygon
from subplots_from_axsize import subplots_from_axsize

from src import jax_utils
from src import jax_evaluation
from src.jax_protocol import Protocol
from src.fig_layout import row_to_pos, val_name_pos_list
from src.internal_abbreviations import format_positive_inhibitors
from src.jax_evaluation import cross_entropy_from_logit
from src.parameters import RESPONSIVENESS_DELAY


PE_COLOR = '#1E88E5'
RE_COLOR = '#D81B60'
MI_COLOR = '#004D40'
MI_NAIVE_COLOR = '#804D40'


inh_to_color = {
    0.: 'blue',
    .3: 'green',
    1.: 'red',
}


BPH = 60 / np.log(2)


def make_title(ax, well_id, protocol_id, model_id, other_info=''):
    ax.set_title(
        f'{well_id}\n'
        f'{protocol_id}; {model_id}; {other_info}'
    )


def plot_predictions_by_ly_mean(
    ax,
    protocol: Protocol,
    predictions,
):
    """Legacy code. Plots TPF & FPR averages as a function of L."""
    predictions['importance_weight'] = predictions.get('importance_weight', 1.0)
    predictions['weighted_p'] = predictions['importance_weight'] * expit(predictions['p_predicted_logit'])
    p_by_ly = predictions.groupby(['L', 'y'])['weighted_p'].sum() / predictions.groupby(['L', 'y'])['importance_weight'].sum()

    mi = jax_evaluation.mutual_information_naive_per_slot_from_predictions(predictions)
    mi_bit_per_hour = mi * jax_utils.BPH

    mi_ce = jax_evaluation.mutual_information_cross_per_slot_from_predictions(predictions)
    mi_ce_bit_per_hour = mi_ce * jax_utils.BPH

    ax.plot(
        p_by_ly.loc[:, 1].index.get_level_values('L') / 60,
        p_by_ly.loc[:, 1],
        '-og',
        label='true positive rate',
        clip_on=False,
    )

    ax.plot(
        p_by_ly.loc[:, 0].index.get_level_values('L') / 60,
        p_by_ly.loc[:, 0],
        '-or',
        label='false positive rate',
        clip_on=False,
    )

    _plot_stopping_prior(ax, protocol)
    ax.legend()

    interval_max = float(protocol.ls.max() / 60)
    ax.set_xlim(0, interval_max)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel('interval [min]')

    ax.annotate(
        f'MI = {mi_bit_per_hour:.1f} bit/hour',
        (interval_max / 2, 0.73),
        va='center',
        ha='center',
        color='black',
        fontsize=16,
    )

    ax.annotate(
        f'MI(c) = {mi_ce_bit_per_hour:.1f} bit/hour',
        (interval_max / 2, 0.63),
        va='center',
        ha='center',
        color='black',
        fontsize=16,
    )

    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(ls=':')


def plot_predictions_by_ly_hist(
    ax,
    protocol: Protocol,
    predictions,
):
    mi = jax_evaluation.mutual_information_naive_per_slot_from_predictions(predictions)
    mi_ce = jax_evaluation.mutual_information_cross_per_slot_from_predictions(predictions)

    rectss = {}
    for (l, y), predictions_ly in predictions.groupby(['L', 'y']):
        n, bins, rects = ax.hist(
            expit(predictions_ly['p_predicted_logit']),
            weights=predictions_ly.get('importance_weight', None),
            density=True,
            color=['maroon', 'limegreen'][y],
            bins=np.linspace(0, 1, 51),
            bottom=l / 60,
            orientation='horizontal',
            alpha=0.7,
        )
        rectss.update({(l, y): rects})


    rects_norm_factor = np.quantile(
        [r.get_width() for (l, y), rects in rectss.items() for r in rects],
        q=0.995,  # some bins are allowed to be "too large", say at L=L_max
    )

    for (l, y), rects in rectss.items():
        for r in rects:
            r.set_width(-0.45 * (-1)**y * r.get_width() / rects_norm_factor)

    _plot_stopping_prior(ax, protocol)

    # add MI
    annotate_kwargs = dict(
        xycoords='axes fraction',
        ha='center',
        font='monospace',
        fontsize='xx-large',
        fontweight='bold',
    )
    ax.annotate(f"MI:    {mi * BPH:5.2f} bit/h", (0.5, 0.70), **annotate_kwargs)
    ax.annotate(f"MI_ce: {mi_ce * BPH:5.2f} bit/h", (0.5, 0.63), **annotate_kwargs)

    ax.set_ylabel('predicted probability of pulse')
    ax.set_xlabel('interval from previous pulse [min]')

    ax.set_xlim(0, float(protocol.ls.max() / 60))
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))


def _plot_stopping_prior(ax, protocol, **kwargs):
    stopping_probs = jnp.exp(protocol.stopping_log_probabilities())

    plot_kwargs = dict(
        color='k',
        ls=':',
        label='stopping prior',
        clip_on=False,
    ) | kwargs

    ax.plot(
        protocol.ls / 60,
        stopping_probs,
        **plot_kwargs,
    )


def _get_predictions_per_track(predictions: pd.DataFrame, protocol: Protocol):
    """
    This function should perhaps compute:
    (prior track entropy - reconstruction track entropy) / track_length
    """
    pe = protocol.entropy_per_slot()

    predictions = predictions.set_index('track_id')

    
    re_ce = cross_entropy_from_logit(predictions['p_predicted_logit'], predictions['y'])
    weighted_re_ce = predictions.get('importance_weight', 1.0) * re_ce

    predictions_per_track = (
        weighted_re_ce.to_frame('re_per_track')
        .groupby('track_id').mean()
    )

    predictions_per_track['mi_per_track'] = pe - predictions_per_track['re_per_track']
    predictions_per_track['mi_per_track_bph'] = predictions_per_track['mi_per_track'] * BPH
    predictions_per_track['track_n_datapoints'] = predictions.groupby('track_id').size()

    return predictions_per_track


def _shuffle(arr):
    shuffled = list(arr)
    random.shuffle(shuffled)
    return shuffled


def plot_mi_per_track(ax, predictions, protocol: Protocol, **kwargs):
    random.seed(1234)
    mock_track_ids = predictions.groupby(['L', 'I'])['track_id'].transform(_shuffle)
    mock_predictions = predictions.drop(columns='track_id').assign(track_id=mock_track_ids)

    predictions_per_track = _get_predictions_per_track(predictions, protocol)
    mock_predictions_per_track = _get_predictions_per_track(mock_predictions, protocol)

    pe = protocol.entropy_per_slot()
    re = (
        (predictions_per_track['re_per_track'] * predictions_per_track['track_n_datapoints']).sum()
      / (predictions_per_track['track_n_datapoints']).sum()
    )
    mi = pe - re

    ax.hist(
        predictions_per_track['re_per_track'] * BPH,
        bins=np.linspace(-0.1 * pe * BPH, 4 * pe * BPH, 101),
        weights=predictions_per_track['track_n_datapoints'],
        density=True,
        **dict(
            color=RE_COLOR,
            alpha=0.7,
            label='RE per track',
        ) | kwargs
    )

    ax.hist(
        mock_predictions_per_track['re_per_track'] * BPH,
        bins=np.linspace(-0.1 * pe * BPH, 4 * pe * BPH, 101),
        weights=mock_predictions_per_track['track_n_datapoints'],
        density=True,
        color='slateblue',
        histtype='step',
        ls=':',
        label='shuffled',
    )
    
    ax.axvline(pe * BPH, color=PE_COLOR, label='sent')
    ax.axvline(re * BPH, color=RE_COLOR, label='lost')
    ax.legend(loc='lower right')
    ax.set_xlabel('reconstruction entropy [bit/h]')

    annotate_kwargs = dict(xycoords='axes fraction', fontsize='xx-large', font='monospace')
    ax.annotate(f"PE: {pe * BPH:5.2f} bit/h", (0.4, 0.8), color=PE_COLOR, **annotate_kwargs)
    ax.annotate(f"RE: {re * BPH:5.2f} bit/h", (0.4, 0.7), color=RE_COLOR, **annotate_kwargs)
    ax.annotate(f"MI: {mi * BPH:5.2f} bit/h", (0.4, 0.6), color=MI_COLOR, **annotate_kwargs)


def plot_correlations_per_track(ax, predictionss, tracks_infos, experiment_ids, fields=None):
    correlations = {}
    for predictions, tracks_info, experiment_id in zip(predictionss, tracks_infos, experiment_ids):

        predictions['re_ce'] = cross_entropy_from_logit(predictions['p_predicted_logit'], predictions['y'])
        predictions['weighted_re_ce'] = predictions.get('importance_weight', 1.0) * predictions['re_ce']
        weighted_log_p_ok_per_track = predictions.groupby('track_id')['weighted_re_ce'].mean()

        predictions_per_track: pd.DataFrame = (
            weighted_log_p_ok_per_track.to_frame('re_per_track')
        )

        predictions_per_track = (
            predictions_per_track
            .join(tracks_info)
            .join(np.log10(tracks_info), rsuffix='_LOG10')
        )
        if fields is not None:
            predictions_per_track = predictions_per_track[fields]

        correlations.update({experiment_id: predictions_per_track.corrwith(weighted_log_p_ok_per_track)})

    correlations = pd.DataFrame(correlations)
    ax.imshow(correlations.to_numpy(), vmin=-1., vmax=1., cmap='RdBu_r')
    ax.set_yticks(np.arange(len(correlations)))
    ax.set_yticklabels(correlations.index)
    ax.set_xticks(np.arange(correlations.shape[1]))
    ax.set_xticklabels(correlations.columns)
    ax.xaxis.set_tick_params(labelrotation=90)


def plot_weariness(ax, data, pulses, well_info, field='log_translocation', shift=RESPONSIVENESS_DELAY):
    mean_per_pulse = data[[f'{field}_ratio', 'time_in_minutes']][data.index.get_level_values('time_in_seconds').isin(pulses)].groupby('time_in_seconds').mean()
    pulses_found = mean_per_pulse.index.to_numpy()
    for it, pulse in enumerate(pulses): 
        if pulse not in pulses_found:
            warnings.warn(f"{it}th pulse (at {pulse}s = {pulse / 60}min) not found in data")
    mean_per_pulse.plot.scatter('time_in_minutes', f'{field}_ratio', c=np.concatenate([np.array([1.2 * np.diff(pulses_found).max()]), np.diff(pulses_found)]) / 60, ax=ax)
    ax.set_ylabel('log translocation ratio [({shift} min : 0 min) after pulse]')

    inhibitor_str = format_positive_inhibitors(well_info, unit_separator=' ',  inhibitors_separator=' + ', no_inh='no inhibitor')

    ax.set_title(
        f'{well_info["experiment"]} '
        f'expo {well_info["exposition"]} '
        f'inh {inhibitor_str} '
        f'rep {well_info["repetition"]}\n'
        f'field, shift={shift}'
    )


def plot_field_scatter(predictions, protocol, tracks_info, field_x, field_y, field_c, seconds_per_timepoint=60, **kwargs):
    predictions_per_track = _get_predictions_per_track(predictions, protocol)
    tracks_info_with_predictions = tracks_info.join(predictions_per_track, how='inner')
    
    return make_scatter(
        tracks_info_with_predictions,
        x=field_x,
        y=field_y,
        c=field_c,
        seconds_per_timepoint=seconds_per_timepoint,
        **kwargs,
    )


def _flip_axis_and_scale(lims, scale=1.5):
    return scale*lims[1], scale*lims[0]


def _add_margin_to_axis(lims, bottom, top):
    return ((1-bottom) * lims[0] + bottom * lims[1],  (1 - top) * lims[0] + top * lims[1])


def make_scatter(tracks_info_with_predictions, x, y, seconds_per_timepoint, c=None, binsx=101, binsy=101, marker_scale=.1, axs=None, ret_fig=True, colorbar=True, **kwargs):
    if axs is None:
        if colorbar:
            fig, axs = subplots_from_axsize(axsize=([1, 6, .3], [6, 1]), right=1.2, top=.7, wspace=.05, hspace=.05)
        else:
            fig, axs = subplots_from_axsize(axsize=([1, 6], [6, 1]), right=1.2, top=.7, wspace=.05, hspace=.05)

    tracks_info_with_predictions = tracks_info_with_predictions.sort_values('track_n_datapoints')
    image = tracks_info_with_predictions.plot.scatter(
        x=x, 
        y=y, 
        s=tracks_info_with_predictions['track_n_datapoints'] * marker_scale, 
        c=tracks_info_with_predictions[c].tolist() if c is not None else None,
        ax=axs[0][1],
        **{
            'cmap': 'rainbow', 
            'alpha': .1, 
        } | kwargs
    )
    if colorbar:
        image.get_lines()
        path_collection = image.get_children()[0]
        plt.colorbar(path_collection, cax=axs[0][2], label=c)
    
    weights = tracks_info_with_predictions['track_n_datapoints'] / (tracks_info_with_predictions['track_end'].max() - tracks_info_with_predictions['track_start'].min() + 60) * seconds_per_timepoint
    axs[1][1].hist(tracks_info_with_predictions[x], bins=binsx, weights=weights, color='grey')
    axs[0][0].hist(tracks_info_with_predictions[y], bins=binsy, weights=weights, color='grey', orientation='horizontal')

    xlim = (tracks_info_with_predictions[x].quantile(.001), tracks_info_with_predictions[x].quantile(.999))
    xlim = _add_margin_to_axis(xlim, -.1, 1.1)
    ylim = (tracks_info_with_predictions[y].quantile(.001), tracks_info_with_predictions[y].quantile(.999))
    ylim = _add_margin_to_axis(ylim, -.1, 1.1)
    axs[0][0].set_ylim(ylim)
    axs[0][1].set_ylim(ylim)
    axs[0][1].set_xlim(xlim)
    axs[1][1].set_xlim(xlim)

    axs[0][0].set_xlim(_flip_axis_and_scale(axs[0][0].get_xlim()))
    axs[1][1].set_ylim(_flip_axis_and_scale(axs[1][1].get_ylim()))

    axs[0][1].yaxis.set_visible(False)
    axs[0][1].xaxis.set_visible(False)
    axs[0][1].set_ylabel('')
    axs[0][1].set_xlabel('')

    axs[0][0].set_ylabel(y)
    axs[1][1].set_xlabel(x)

    axs[1][0].set_visible(False)
    if colorbar:
        axs[1][2].set_visible(False)

    if ret_fig:
        return fig, axs[0][1]


def plot_response_amplitude_by_interval(ax, tracks_preprocessed, seconds_per_timepoint):
    tracks_preprocessed['L'] = tracks_preprocessed['L'] // 60
    tracks_preprocessed['I'] = tracks_preprocessed['I'] // 60

    used_timepoints = (
        (
            tracks_preprocessed['y'] 
          | (tracks_preprocessed['I'] - tracks_preprocessed['L'] >= RESPONSIVENESS_DELAY * (60 // seconds_per_timepoint)) 
        )
      & tracks_preprocessed['translocation'].gt(0) 
      & tracks_preprocessed['I'].notna()
    )


    rectss = {}
    for (l, y), results_ly in tracks_preprocessed[used_timepoints].groupby(['L', 'y']):
        n, bins, rects = ax.hist(
            np.exp(results_ly['log_translocation_ratio']),
            # weights=results_ly['importance_weight'] / results_ly['importance_weight'].sum(),
            density=True,
            color=['maroon', 'limegreen'][y],
            bins=np.linspace(0, 2.5, 51),
            bottom=l,
            orientation='horizontal',
            alpha=0.7,
            # edgecolor='gray',
            # lw=.5,
        )
        rectss.update({(l, y): rects})

    n, bins, rects = ax.hist(
        np.exp(tracks_preprocessed[used_timepoints & (tracks_preprocessed['y'] == 1)]['log_translocation_ratio']),
        # weights=results_ly['importance_weight'] / results_ly['importance_weight'].sum(),
        density=True,
        color=['maroon'],
        bins=np.linspace(0, 2.5, 51),
        bottom=0,
        orientation='horizontal',
        alpha=0.7,
        # edgecolor='gray',
        # lw=.5,
    )
    rectss.update({(0, 0): rects})


    rects_norm_factor = np.quantile(
        [r.get_width() for (l, y), rects in rectss.items() for r in rects],
        q=0.995,  # some bins are allowed to be "too large", say at L=L_max
    )

    for (l, y), rects in rectss.items():
        for r in rects:
            r.set_width(-0.45 * (-1)**y * r.get_width() / rects_norm_factor)

    mean_response = tracks_preprocessed[used_timepoints].groupby(['L', 'y'])['log_translocation_ratio'].mean()
    mean_response.loc[0, 0] = tracks_preprocessed[used_timepoints & (tracks_preprocessed['y'] == 1)]['log_translocation_ratio'].mean()
    mean_response.sort_index().pipe(np.exp).unstack('y').reindex(columns=[1, 0]).plot(ax=ax, marker='x', color=['green', 'red'])
    ax.legend(['response (pulses at 0 and $t$)', 'reference (pulse only at 0)'])
    ax.set_xlim(-1, 31)
    ax.set_ylabel(f'translocation fold-change during next {RESPONSIVENESS_DELAY} min')
    ax.set_xlabel('time $t$ [min]')
    ax.grid(axis='x', ls=':', color='k', alpha=.3)



def plot_log_response_over_reference_by_interval(ax, response_over_reference, **kwargs):
    response_over_reference.index = np.array(response_over_reference.index) / 60
    response_over_reference.plot(ax=ax, **dict(marker='x', color='goldenrod') | kwargs)

    # ax.legend(['response (pulses at 0 and $t$)', 'reference (pulse only at 0)'])
    ax.set_xlim(-1, 31)
    ax.set_ylim(-.3, .8)
    ax.set_ylabel('average response amplitude\n[log fold change]')
    ax.set_xlabel('time $t$ [min]')
    ax.grid(ls=':', color='k', alpha=.3)


# TODO(frdrc): Legacy. Not working. Should be revived.
# def plot_tpr_violin_trajectory(ax, pulses, predictions):
#     lis = [
#         (l, i // 60, t_last + l * 60)
#         for t_last, i in zip(pulses, np.diff(pulses))
#         for l in range(1, i // 60 + 1)
#     ]
# 
#     li_index = pd.DataFrame(lis, columns=['L', 'I', 'time']).set_index(['L', 'I'])['time']
# 
#     predictions_by_time_point = predictions.join(li_index, on=['L', 'I']).sort_values('time').reset_index()
# 
#     sns.violinplot(
#         predictions_by_time_point,
#         x='time',
#         y='p',
#         inner='quart',
#         native_scale=True,
#         cut=0,
#         density_norm='width',
#         ax=ax,
#     )


# TODO(frdrc): Currenlty unused, but can be adopted to
# scan arbitrary protocol list with a single model
# def compute_mi_vs_alpha(model, dataloader, alphas):
#     
#     intervals = dataloader.intervals
#     protocol_template = geometric_protocol(alphas[0], intervals)
#     batch = dataloader.get_full_dataset_as_batch(protocol_template)
# 
#     mis = []
#     mis_ce = []
#     for alpha in alphas:
#         protocol = geometric_protocol(alpha, intervals)
# 
#         batch_ps = model(protocol, batch)
#         li_logprobs = protocol.li_joint_log_probabilities()
#         batch_log_ws = li_logprobs[batch['l_idxs'], batch['i_idxs']] - batch['sampling_li_logprobs']
#         batch_ws = jnp.exp(batch_log_ws)
# 
#         res_naive = jax_utils.bi_entropy(batch_ps)
#         res_cross = np.log(batch_ps * batch['ys'] + (1 - batch_ps) * (1 - batch['ys']))
# 
#         re_naive = (res_naive * batch_ws).mean()
#         re_cross = (res_cross * batch_ws).mean()
# 
#         pe = protocol.entropy_per_slot()
# 
#         mi_naive = pe - re_naive
#         mi_naive_bit_per_hour = mi_naive * jax_utils.BPH
# 
#         mi_cross = pe - re_cross
#         mi_cross_bit_per_hour = mi_cross * jax_utils.BPH
# 
#         mis.append(mi_naive_bit_per_hour)
#         mis_ce.append(mi_cross_bit_per_hour)
#     
#     return pd.DataFrame({
#        'alpha': alphas,
#        'mi': mis,
#        'mi_ce': mis_ce,
#    }).set_index('alpha')



def plot_mi(ax, mi_all: pd.DataFrame, field, title, ymin=0, ymax=-np.inf, bar_height=.3, plot_labels=True, annotate_means=True, tick_means=False, means_format_str='{mean}', seed=0, **kwargs):
    random_state = RandomState(MT19937(SeedSequence(seed)))
    if 'pos' not in mi_all:
        mi_all['pos'] = mi_all.apply(row_to_pos, axis=1)
    
    ymin = min(ymin, 1.1 * mi_all[field].min())
    ymax = max(ymax, 1.1 * mi_all[field].max())

    ax2_ticks = []
    ax2_tickslabels = []

    for pos, mi_pos in mi_all.groupby('pos'):
        mean = mi_pos[field].mean()
        sem = mi_pos[field].sem()
        ax.add_patch(Rectangle((ymin, pos - bar_height / 2), width=ymax-ymin, height=bar_height, color=(.8, .8, .8)))
        ax.add_patch(Rectangle((mean - sem, pos - bar_height / 2), width=2*sem, height=bar_height, color='red', alpha=.2))
        ax.vlines([mean], pos - bar_height/2, pos + bar_height/2, ls='-', color='red')
        ax.vlines([mean - sem, mean + sem], pos - bar_height / 2, pos + bar_height / 2, ls='-', color='red', alpha=.3)
        if annotate_means:
            ax.annotate(means_format_str.format(mean=mean, sem=sem), ((ymin + ymax) / 2, pos - bar_height / 2),  horizontalalignment='center', verticalalignment='bottom')
        if tick_means:
            ax2_ticks.append(pos)
            ax2_tickslabels.append(means_format_str.format(mean=mean, sem=sem))

    for experiment, mi_plot_exp in mi_all.groupby('experiment'):
        ax.plot(
            mi_plot_exp[field],
            add_noise(mi_plot_exp['pos'], bar_height / 3, random_state=random_state),
            ls='none',
            label=experiment,
            **dict(
                marker='o',
                ms=2,
                alpha=.7,
            ) | kwargs,
        )

    # ax.set_ylim(10.0, .5)
    ax.set_ylim(mi_all['pos'].max() + .5, mi_all['pos'].min() - .5)
    ax.set_xlim(ymin, ymax) # note that x any y are swapped
    # ax.grid(color='k', alpha=0.5, ls=':')
    ax.set_xlabel(title)
    if plot_labels:
        ax.set_yticks(*list(zip(*[(pos, name) for _, name, pos in val_name_pos_list])))
    else:
        ax.set_yticks([])

    ax.spines[['top', 'right', 'left']].set_visible(False)


    if tick_means:
        means = mi_all.groupby('pos')[field].mean()
        sems = mi_all.groupby('pos')[field].sem()
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(means.index, [means_format_str.format(mean=mean, sem=sem) for mean, sem in zip(means, sems)])
        ax2.spines[['top', 'right', 'left']].set_visible(False)


def plot_mi_diff(ax, mi_all_1: pd.DataFrame, mi_all_2: pd.DataFrame, field, title, ymin=0, ymax=-np.inf, bar_height=.3, plot_labels=True, annotate_means=True, tick_means=False, means_format_str='{mean}', index_cols=None, **kwargs):
    if index_cols:
        mi_all = mi_all_1.join(mi_all_2.reset_index().set_index(index_cols)[field], on=index_cols, lsuffix='_1', rsuffix='_2')
    else:
        mi_all = mi_all_1.join(mi_all_2[field], on=index_cols, lsuffix='_1', rsuffix='_2')

    
    if 'pos' not in mi_all:
        mi_all['pos'] = mi_all.apply(row_to_pos, axis=1)

    ymin = min(ymin, 1.1 * mi_all[[f"{field}_1", f"{field}_2"]].min().min())
    ymax = max(ymax, 1.1 * mi_all[[f"{field}_1", f"{field}_2"]].max().max())
    
    ax2_ticks = []
    ax2_tickslabels = []

    for pos, mi_pos in mi_all.groupby('pos'):
        mean_1 = mi_pos[f"{field}_1"].mean()
        sem_1 = mi_pos[f"{field}_1"].sem()
        mean_2 = mi_pos[f"{field}_2"].mean()
        sem_2 = mi_pos[f"{field}_2"].sem()
        # print(mean_1, sem_1, mean_2, sem_2)
        ax.add_patch(Rectangle((ymin, pos - bar_height / 2), width=ymax-ymin, height=bar_height, color=(.8, .8, .8)))
        ax.add_patch(Polygon(
            [
                (mean_1, pos - bar_height / 2),
                (1. * mean_1 + .0 * mean_2, pos - bar_height / 2),
                (mean_2, pos),
                (1. * mean_1 + .0 * mean_2, pos + bar_height / 2),
                (mean_1, pos + bar_height / 2),
            ], 
            closed=True, 
            color='green' if mean_2 > mean_1 else 'red',
            ))
        # ax.vlines([mean], pos - bar_height/2, pos + bar_height/2, ls='-', color='red')
        # ax.vlines([mean - sem, mean + sem], pos - bar_height / 2, pos + bar_height / 2, ls='-', color='red', alpha=.3)
        if annotate_means:
            ax.annotate(means_format_str.format(mean_1=mean_1, sem_1=sem_1, mean_2=mean_2, sem_2=sem_2), ((ymin + ymax) / 2, pos - bar_height / 2),  horizontalalignment='center', verticalalignment='bottom')
        if tick_means:
            ax2_ticks.append(pos)
            ax2_tickslabels.append(means_format_str.format(mean_1=mean_1, sem_1=sem_1, mean_2=mean_2, sem_2=sem_2))

    # ax.set_ylim(10.0, .5)
    ax.set_ylim(mi_all['pos'].max() + .5, mi_all['pos'].min() - .5)
    ax.set_xlim(ymin, ymax) # note that x any y are swapped
    # ax.grid(color='k', alpha=0.5, ls=':')
    ax.set_xlabel(title)
    if plot_labels:
        ax.set_yticks(*list(zip(*[(pos, name) for _, name, pos in val_name_pos_list if pos in mi_all['pos'].unique()])))
    else:
        ax.set_yticks([])

    ax.spines[['top', 'right', 'left']].set_visible(False)

    if tick_means:
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(ax2_ticks, ax2_tickslabels)
        ax2.spines[['top', 'right', 'left']].set_visible(False)


def add_noise(x, scale=1, *, random_state: RandomState):
    return x + 2 * scale * (random_state.random(size=x.size) - .5)


