import numpy as np
import pandas as pd

from typing import Optional

from src.data_preprocessing import tracks_transform_column

pd.set_option('future.no_silent_downcasting', True)


def create_dataset(
    tracks: pd.DataFrame,
    tracks_info: pd.DataFrame,
    col='translocation',
    log_transform=True,
    differentiate=True,
    rolling_min_window=None,
    add_log_l=True,
    extra_cols=[],
    quality=0,
    tracks_mi: Optional[pd.DataFrame] = None, # used only if quality == 'transmitting'
    r_ts=[60, 120, 180, 240, 300, 360],
    seconds_per_timepoint=60,
    well_id=None,
) -> pd.DataFrame:
    r_ts = np.array(r_ts)

    tracks = tracks[tracks['valid']].copy()

    # filter by quality
    tracks = tracks.join(
        tracks_info['quality'],
        on='track_id',
    )
    if quality == 'transmitting':
        assert tracks_mi is not None
        tracks = tracks[tracks.join(tracks_mi, on='track_id')['is_transmitting'].fillna(False).astype(bool)]
    else:
        tracks = tracks[tracks['quality'] >= quality]
    
    tracks[col] = tracks_transform_column(
        tracks,
        col=col,
        differentiate=differentiate,
        log_transform=log_transform,
        rolling_min_window=rolling_min_window,
        seconds_per_timepoint=seconds_per_timepoint,
    )

    dataset = _create_dataset_responses(
        tracks=tracks,
        r_ts=r_ts,
        col=col,
        seconds_per_timepoint=seconds_per_timepoint,
    )

    if add_log_l:
        log_ls = np.log(dataset['L'])
        dataset[f'X_log_l'] = log_ls
    dataset = dataset.join(
        tracks_info[extra_cols],
        on='track_id',
    )
    
    if well_id:
        dataset['well_id'] = well_id

    # drop nans
    dataset = dataset.dropna()

    return dataset


def _create_dataset_responses(
    tracks,
    col,
    r_ts,
    seconds_per_timepoint,
):
    shifts = -r_ts / seconds_per_timepoint
    assert np.all(shifts == shifts.round())
    shifts = shifts.astype(int)

    dataset = tracks[['L', 'I']].copy()
    response = tracks[col]

    for dt, shift in zip(r_ts, shifts):
        dataset[f'X{dt:+d}'] = response.groupby(level='track_id').shift(shift)

    dataset = dataset.reset_index()

    dataset = dataset.dropna()
    dataset['L'] = dataset['L'].astype(int)
    dataset['I'] = dataset['I'].astype(int)

    return dataset

