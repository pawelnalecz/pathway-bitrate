from pathlib import Path
import sys
sys.path.append(str(Path('.').resolve()))

import pandas as pd
import numpy as np

from config.parameters import RESPONSIVENESS_DELAY
from src.data_preprocessing import compute_translocation_ratio

# AUXILIARY FUNCTIONS

def get_tracks_preprocessed_with_quality(wildcards, input):
    per_well = wildcards.per == 'per_well'
    well_ids = [wildcards.well_or_set_id] if per_well else SET_ID_TO_WELL_IDS[wildcards.well_or_set_id]
    tracks_preprocessed = pd.read_pickle(str(input.tracks_preprocessed[0])) if per_well else pd.concat(
        (pd.read_pickle(tracks_path) for tracks_path in input.tracks_preprocessed),
        names=['well_id'],
        keys=well_ids,
    ) 
    experiments = [DATA_MANAGER.get_experiment(well_id) for well_id in well_ids]
    seconds_per_timepoint = DATA_MANAGER.get_seconds_per_timepoint_for_experiment_list(experiments)
    index_col = 'track_id' if per_well else ['well_id', 'track_id']
    tracks_info = pd.read_csv(str(input.tracks_info), index_col=index_col)

    quality = int(wildcards.quality)
    tracks_preprocessed = tracks_preprocessed[tracks_preprocessed.join(tracks_info['quality'], on=index_col)['quality'] >= quality]
    return tracks_preprocessed, seconds_per_timepoint


# RULES

rule measures:
    input:
        tracks_preprocessed=inputs_tracks_preprocessed,
        tracks_info=inputs_tracks_info,
        response_reference=f'cache/response_amplitude/single/per_well/{{well_id}}/response-reference_q{{quality}}_{RESPONSIVENESS_DELAY}.csv'
    output:
        'cache/measures/per_well/{well_id}/measures_q{quality}.csv'
    resources:
        mem_gib=1
    run:
        well_id = wildcards.well_id
        well_info = DATA_MANAGER.get_well_info(well_id)

        experiment = DATA_MANAGER.get_experiment(well_id)
        experiment_info = DATA_MANAGER.get_experiment_info(experiment)
        seconds_per_timepoint = DATA_MANAGER.get_seconds_per_timepoint(experiment)

        pulses = DATA_MANAGER.get_pulses(experiment)['time_in_seconds']

        quality = int(wildcards.quality)
        tracks_info = pd.read_csv(input.tracks_info, index_col='track_id')
        tracks = pd.read_pickle(input.tracks_preprocessed)
        tracks = tracks[tracks.join(tracks_info, on='track_id')['quality'] >= quality]

        response_reference = pd.read_csv(input.response_reference).set_index('L')

        average_response_amplitude = tracks['log_translocation_ratio'][tracks['y'] == 1].mean()
        average_response_amplitude_over_reference = (
            tracks
            .join(response_reference, on='L')
            [tracks['y'] == 1]
            .pipe(lambda x: x['log_translocation_ratio'] - x['response_reference'])
            .mean()
        )
        first_pulse_response = tracks['log_translocation_ratio'][tracks.index.get_level_values('time_in_seconds') == pulses[0]].mean()

        measures_df = pd.DataFrame({
            'well_id': well_id,
            'cell_line': experiment_info['cell_line'],
            **well_info,
            'average_response_amplitude': average_response_amplitude,
            'average_response_amplitude_over_reference': average_response_amplitude_over_reference,
            'first_pulse_response': first_pulse_response,
        }, index=[0])
        measures_df.to_csv(str(output), index=False)


rule response_reference:
    input:
        tracks_preprocessed=inputs_tracks_preprocessed,
        tracks_info=inputs_tracks_info,
    output:
        'cache/response_amplitude/single/{per}/{well_or_set_id}/response-reference_q{quality}_{delay}.csv'
    resources:
        mem_gib=lambda wc, input: len(input)
    run:
        tracks_preprocessed, seconds_per_timepoint = get_tracks_preprocessed_with_quality(wildcards, input)
        delay = int(wildcards.delay)

        valid_timepoints = (
            tracks_preprocessed['translocation'].gt(0) 
          & tracks_preprocessed['I'].notna()
        )

        long_before_pulse = (
            valid_timepoints
          & (tracks_preprocessed['y'] == 0)
          & (tracks_preprocessed['I'] - tracks_preprocessed['L'] > delay * 60) 
        )

        at_pulse = (
            valid_timepoints
          & (tracks_preprocessed['y'] == 1)
        )

        if delay != RESPONSIVENESS_DELAY:    
            with np.errstate(divide='ignore'):
                tracks_preprocessed['log_translocation'] = np.log(tracks_preprocessed['translocation']).replace(-np.inf, np.nan)
            compute_translocation_ratio(tracks_preprocessed, fields=['log_translocation'], shift_in_minutes=delay, seconds_per_timepoint=seconds_per_timepoint)


        log_response_by_L = tracks_preprocessed[long_before_pulse].groupby('L')['log_translocation_ratio'].mean()
        log_response_by_L.loc[0] = tracks_preprocessed[at_pulse]['log_translocation_ratio'].mean()
        log_response_by_L.name = 'response_reference'
        log_response_by_L.index = log_response_by_L.index.astype(int)

        log_response_by_L.sort_index().to_csv(str(output))


rule response_over_reference:
    input:
        tracks_preprocessed=inputs_tracks_preprocessed,
        tracks_info=inputs_tracks_info,
        response_reference='cache/response_amplitude/single/{per}/{well_or_set_id}/response-reference_q{quality}_{delay}.csv',
    output:
        'cache/response_amplitude/single/{per}/{well_or_set_id}/response-over-reference_q{quality}_{delay}.csv'
    resources:
        mem_gib=lambda wc, input: len(input)
    run:
        tracks_preprocessed, seconds_per_timepoint = get_tracks_preprocessed_with_quality(wildcards, input)
        log_response_reference = pd.read_csv(input.response_reference, index_col=['L'])['response_reference']
        delay = int(wildcards.delay)
            
        valid_timepoints = (
            tracks_preprocessed['translocation'].gt(0) 
          & tracks_preprocessed['I'].notna()
        )

        at_pulse = (
            valid_timepoints
          & (tracks_preprocessed['y'] == 1)
        )

        if delay != RESPONSIVENESS_DELAY:    
            with np.errstate(divide='ignore'):
                tracks_preprocessed['log_translocation'] = np.log(tracks_preprocessed['translocation']).replace(-np.inf, np.nan)
            compute_translocation_ratio(tracks_preprocessed, fields=['log_translocation'], shift_in_minutes=delay, seconds_per_timepoint=seconds_per_timepoint)


        log_response_by_L = tracks_preprocessed[at_pulse].groupby('L')['log_translocation_ratio'].mean()
        log_response_by_L_normalized = log_response_by_L - log_response_reference
        log_response_by_L_normalized.name = 'response_amplitude'

        log_response_by_L_normalized.sort_index().to_csv(str(output))

