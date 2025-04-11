from typing import Optional

import pandas as pd
import numpy as np

from src.timer import Timer
from config import parameters
from src.stat_utils import normalize_with_mean_and_std
from src.internal_abbreviations import has_exactly_inhibitors

DEBUG = False

# TODO(frdrc): double check this


def preprocess_tracks(
    tracks,
    pulses,
    experiment_info,
    remove_short_tracks=True,
    background_clip=0.03,
    #smoothing_wins=range(1, 10),
):
    timer = Timer()
    timer.start()

    receptor_channel  = experiment_info['receptor_channel']
    reporter_channel  = experiment_info['reporter_channel']

    valid_pulses = np.array(pulses[pulses['valid']]['time_in_seconds'])

    _add_time_after_pulse(tracks, valid_pulses, inplace=True)
    
    if remove_short_tracks: 
        tracks = _remove_short_tracks(tracks, valid_pulses)
        timer.log('Removing short tracks')

    tracks[f'img_{receptor_channel}_background'] = tracks[f'img_{receptor_channel}_background'].replace(0, np.nan)
    tracks[f'nuc_{receptor_channel}_intensity_mean'] = tracks[f'nuc_{receptor_channel}_intensity_mean'].replace(0, np.nan)

    use_manual_receptor = f'{receptor_channel}_background' in experiment_info and not np.isnan(experiment_info[f'{receptor_channel}_background'])
    use_manual_reporter = f'{reporter_channel}_background' in experiment_info and not np.isnan(experiment_info[f'{reporter_channel}_background'])
    
    if use_manual_receptor:
        print(f'Using manually adjusted {receptor_channel}_background')
    if use_manual_reporter:
        print(f'Using manually adjusted {reporter_channel}_background')

    receptor_background = experiment_info[f'{receptor_channel}_background'] if use_manual_receptor else tracks[f'img_{receptor_channel}_background']
    reporter_background = experiment_info[f'{reporter_channel}_background'] if use_manual_reporter else tracks[f'img_{reporter_channel}_background']

    tracks['receptor_background'] = receptor_background
    tracks['reporter_background'] = reporter_background

    tracks['receptor'] = (tracks[f'nuc_{receptor_channel}_intensity_mean'] - receptor_background).clip(background_clip * receptor_background) # clip necessary for numerical reasons
    tracks['reporter'] = (tracks[f'nuc_{reporter_channel}_intensity_mean'] - reporter_background).clip(background_clip * reporter_background) # clip necessary for numerical reasons
    tracks['translocation'] = (tracks[f'img_{reporter_channel}_intensity_mean'] - reporter_background) / (tracks[f'nuc_{reporter_channel}_intensity_mean'] - reporter_background).clip(background_clip * reporter_background)
    
    if DEBUG:
        is_receptor_below_bg = tracks[f'nuc_{receptor_channel}_intensity_mean'] <= receptor_background
        is_reporter_below_bg = tracks[f'nuc_{reporter_channel}_intensity_mean'] <= reporter_background

        has_any_receptor_below_bg = is_receptor_below_bg.groupby('track_id').any()
        has_any_reporter_below_bg = is_reporter_below_bg.groupby('track_id').any()
        track_length = tracks.groupby('track_id').size()

        print(f"{experiment_info.name}: Timepoints with receptor below threshold: {is_receptor_below_bg.sum() / len(tracks):.2g} timepoints ({(has_any_receptor_below_bg * track_length).sum() / track_length.sum()} of tracks)")
        print(f"{experiment_info.name}: Timepoints with reporter below threshold: {is_reporter_below_bg.sum() / len(tracks):.2g} timepoints ({(has_any_reporter_below_bg * track_length).sum() / track_length.sum()} of tracks)")

    with np.errstate(divide='ignore'):
        tracks['log_translocation'] = np.log(tracks['translocation']).replace(-np.inf, np.nan)
    compute_translocation_ratio(tracks, fields=['log_translocation'], shift_in_minutes=parameters.RESPONSIVENESS_DELAY, seconds_per_timepoint=experiment_info['seconds_per_timepoint'])


    tracks = tracks[[
        'L',
        'I',
        'y',
        'translocation',
        'receptor',
        'reporter',
        # f'nuc_{reporter_channel}_intensity_max',
        # f'log_nuc_{receptor_channel}_intensity_mean',
        # f'log_nuc_{reporter_channel}_intensity_mean',
        f'nuc_{receptor_channel}_intensity_mean',
        f'nuc_{reporter_channel}_intensity_mean',
        'receptor_background',
        'reporter_background',
        'log_translocation_ratio',
        'valid',
    ]]

    timer.log()
    timer.report()
    return tracks



def create_tracks_info(tracks: pd.DataFrame, pulses: pd.DataFrame, well_info, experiment_info):
    tracks = tracks[tracks['valid']].copy()
    
    receptor_channel  = experiment_info['receptor_channel']
    reporter_channel  = experiment_info['reporter_channel']

    with np.errstate(divide='ignore'):
        tracks['log_receptor'] = np.log(tracks['receptor']).replace(-np.inf, np.nan)
        tracks['log_reporter'] = np.log(tracks['reporter']).replace(-np.inf, np.nan)

    valid_pulses = np.array(pulses[pulses['valid']]['time_in_seconds'])

    tracks_info_parts = []
    for track_id, track in tracks.groupby('track_id'):
        responsiveness = track['log_translocation_ratio'][(track['y'] == 1)].mean()
        potential_reposniveness = track['log_translocation_ratio'].std()

        response_reference = track[track['I'] - track['L'] > parameters.RESPONSIVENESS_DELAY].groupby('L')['log_translocation_ratio'].mean()
        response_reference.name = 'response_reference'
        average_response_amplitude_over_reference = (
            track
            .join(response_reference, on='L')
            [track['y'] == 1]
            .pipe(lambda x: x['log_translocation_ratio'] - x['response_reference'])
            .mean()
        )

        tracks_info_parts.append({
            'track_id': track_id,
            'track_start': track.index.get_level_values('time_in_seconds').min(),
            'track_end': track.index.get_level_values('time_in_seconds').max(),
            'track_length': len(track),
            # 'is_overexposed': (track['nuc_ERKKTR_intensity_max'] == 255).any(),
            'is_receptor_below_background': (track[f'nuc_{receptor_channel}_intensity_mean'] <= track['receptor_background']).any(),
            'is_reporter_below_background': (track[f'nuc_{reporter_channel}_intensity_mean'] <= track['reporter_background']).any(),
            'responsiveness': responsiveness,
            'potential_responsiveness': potential_reposniveness,
            'average_response_amplitude_over_reference': average_response_amplitude_over_reference,

        })


    tracks_info = (
        tracks.groupby('track_id').mean()
        .join(tracks.groupby('track_id').std(), lsuffix='_MEAN', rsuffix='_STD')
        .join(pd.DataFrame(tracks_info_parts).set_index('track_id'))
    )


    tracks_info['log_receptor_normalized'] = normalize_with_mean_and_std(tracks_info['log_receptor_MEAN'], tracks_info['track_length'])
    tracks_info['log_reporter_normalized'] = normalize_with_mean_and_std(tracks_info['log_reporter_MEAN'], tracks_info['track_length'])


    # Add fields used in training
    tracks_info = tracks_info.join(
        tracks['log_translocation_ratio']
        .groupby('track_id')
        .std()
        .to_frame('X_ratio_std')
    )
    
    tracks_info = tracks_info.join(
        tracks['log_translocation_ratio']
        [tracks.index.get_level_values('time_in_seconds').isin(valid_pulses)]
        .groupby('track_id')
        .mean()
        .to_frame('X_responsiveness_old')
    )
    

    tracks_info['X_responsiveness'] = tracks_info['responsiveness']
    tracks_info['X_potential_responsiveness'] = tracks_info['potential_responsiveness']

    is_ste1 = well_info['cell_line'] == 'STE1'
    tracks_info['X_cell_line'] = is_ste1

    for inh in ['crizotinib', 'trametinib', 'cyclosporin']:
        tracks_info[f'X_{inh[:4]}'] = well_info[f"inh_{inh}"] #np.nan_to_num(well_info.get(inhibitor_name, default=np.nan), 0.)

    tracks_info['X_cond_STE1_0uM']  = is_ste1 and has_exactly_inhibitors(well_info, [])
    tracks_info['X_cond_STE1_03uM'] = is_ste1 and has_exactly_inhibitors(well_info, 'crizotinib', .3)
    tracks_info['X_cond_STE1_1uM']  = is_ste1 and has_exactly_inhibitors(well_info, 'crizotinib', 1.)
    tracks_info['X_cond_STE1_3uM']  = is_ste1 and has_exactly_inhibitors(well_info, 'crizotinib', 3.)

    tracks_info['X_cond_BEAS2B_0uM']  = not is_ste1 and has_exactly_inhibitors(well_info, [])
    tracks_info['X_cond_BEAS2B_criz'] = not is_ste1 and has_exactly_inhibitors(well_info, 'crizotinib')
    tracks_info['X_cond_BEAS2B_tram'] = not is_ste1 and has_exactly_inhibitors(well_info, 'trametinib')
    tracks_info['X_cond_BEAS2B_cycl'] = not is_ste1 and has_exactly_inhibitors(well_info, 'cyclosporin')
    tracks_info['X_cond_BEAS2B_trcy'] = not is_ste1 and has_exactly_inhibitors(well_info, ['trametinib', 'cyclosporin'])

    # Make sure all X columns are floats
    for col in tracks_info:
        if col.startswith('X_'):
            tracks_info[col] = tracks_info[col].astype(float)

    # Track quality assessment
    # Q0 = {long enough tracks}
    is_Q0 = (
        (tracks_info['track_length'] >= parameters.TRACK_LENGTH_THR)
    )

    # Q1 = {receptor in range}
    is_Q1 = (
        is_Q0
      & tracks_info['log_receptor_normalized'].between(
          experiment_info[f'receptor_lower_thr'],
          experiment_info[f'receptor_upper_thr'],
        )
    )

    # Q2 = {reporter in range}
    is_Q2 = (
        is_Q1
      & tracks_info['log_reporter_normalized'].between(
          experiment_info[f'reporter_lower_thr'],
          experiment_info[f'reporter_upper_thr'],
        )
    )

    # Q3 = {responding tracks}
    is_Q3 = (
        is_Q2
      & (tracks_info['responsiveness'] >= parameters.RESPONSIVENESS_THR)
    )
    tracks_info['quality'] = -1 + is_Q0 + is_Q1 + is_Q2 + is_Q3

    # assert (tracks_info['quality'] > 0).any(), "All tracks have quality<=0!"

    return tracks_info



def tracks_transform_column(
    tracks,
    col,
    differentiate: bool = False,
    log_transform: bool = False,
    rolling_min_window: Optional[int] = None,
    seconds_per_timepoint: int = 60,
):
    """Apply given per-track operations to specified columns."""
    tracks_field = tracks[col]

    if log_transform:
        tracks_field = np.log(tracks_field)

    if rolling_min_window:
        tracks_field = _tracks_rolling_min(tracks_field, rolling_min_window, seconds_per_timepoint)

    if differentiate:
        tracks_field = _tracks_derivative(tracks_field, seconds_per_timepoint)

    return tracks_field



def compute_translocation_ratio(df, fields, shift_in_minutes, seconds_per_timepoint, by='track_id', inplace=True):
    shifted = df.groupby(by)[fields].shift(-shift_in_minutes * 60 // seconds_per_timepoint)
    result_df = shifted - df[fields]
    if inplace:
        mutated_fields = [field + '_ratio' for field in fields]
        df[mutated_fields] = result_df.to_numpy()
    else:
        return result_df




def _tracks_rolling_min(tracks_field: pd.Series, window_in_min: int, seconds_per_timepoint: int):
    """Return minimum in a rolling window of length window_in_min in each track separately."""
    assert tracks_field.index.names == ('track_id', 'time_in_seconds')
    window_in_slots = int(1 + (window_in_min - 1) * (60 / seconds_per_timepoint))
    return (
        tracks_field
        .groupby(level='track_id')
        .rolling(window_in_slots, center=True)
        .min()
        .reset_index(level=0, drop=True)  # avoid duplicated 'track_id' index
    )


def _tracks_derivative(tracks_field: pd.Series, seconds_per_timepoint: int, step_in_seconds: int = 60):
    """Always returns increase in step_in_seconds, regardless of sampling frequency.
    seconds_per_timepoint must be a divisor of step_in_seconds."""
    assert tracks_field.index.names == ('track_id', 'time_in_seconds')
    shift = int(step_in_seconds / seconds_per_timepoint)
    tracks_field_previous = tracks_field.groupby(level='track_id').shift(shift)
    return tracks_field - tracks_field_previous


def _add_time_after_pulse(data: pd.DataFrame, pulses: np.ndarray, inplace=True):
    if not inplace:
        data = data.copy()

    time_in_seconds = data.index.get_level_values('time_in_seconds').unique()
    time_in_seconds_to_timepoint = pd.Series(np.arange(len(time_in_seconds)), index=time_in_seconds)
    
    intervals = np.diff(pulses)

    last_pulse_ids = np.array([
        max(pulse_id for pulse_id, pulse in enumerate(pulses) if pulse < tp) if tp > pulses[0] else -1
        for tp in time_in_seconds
    ])
    last_pulse = np.where(last_pulse_ids >= 0, pulses.take(last_pulse_ids, mode='clip'), np.nan)
    if len(pulses) > 1:
        last_interval = np.where(last_pulse_ids >= 1, intervals.take(last_pulse_ids - 1, mode='clip'), np.nan)
        this_interval = np.where((last_pulse_ids >= 0) & (last_pulse_ids < len(pulses) - 1), intervals.take(last_pulse_ids, mode='clip'), np.nan)
    else:
        last_interval = np.nan * np.ones_like(last_pulse_ids)
        this_interval = np.nan * np.ones_like(last_pulse_ids)
    
    data['last_pulse_id'] = last_pulse_ids[time_in_seconds_to_timepoint[data.index.get_level_values('time_in_seconds')]]
    data['last_pulse'] = last_pulse[time_in_seconds_to_timepoint[data.index.get_level_values('time_in_seconds')]]
    data['last_interval'] = last_interval[time_in_seconds_to_timepoint[data.index.get_level_values('time_in_seconds')]]
    data['I'] = this_interval[time_in_seconds_to_timepoint[data.index.get_level_values('time_in_seconds')]]
    data['L'] = data.index.get_level_values('time_in_seconds') - data['last_pulse']
    data['y'] = 1 * data.index.isin(pulses, level='time_in_seconds')

    if not inplace:
        return data


def _remove_short_tracks(data, valid_pulses, thr_in_min=parameters.TRACK_LENGTH_THR):
    """Removes tracks whose overlap with the interval between the first and the last valid pulse is shorter than thr_in_min."""
    earliest_track_end = valid_pulses[0] + thr_in_min * 60
    latest_track_start = valid_pulses[-1] - thr_in_min * 60
    timer = Timer(prefix='sh  ')
    timer.start()
    track_lens = data.groupby('track_id').size()
    track_start = data.reset_index('time_in_seconds').groupby('track_id')['time_in_seconds'].min()
    track_end = data.reset_index('time_in_seconds').groupby('track_id')['time_in_seconds'].max()
    print(f"Removing {(track_lens < thr_in_min).sum()} tracks shorter than {thr_in_min} timepoints.")
    timer.log('groupby')
    
    tracks_nice = track_lens[
        (track_lens >= thr_in_min) 
        & (track_end >= earliest_track_end)
        & (track_start <= latest_track_start)
        ].index

    if len(tracks_nice) < len(data.index.get_level_values('track_id').unique()):
        data = data.loc[tracks_nice]
    timer.log('allocation')
    timer.report()
    return data

