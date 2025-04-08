import numpy as np
import pandas as pd

from typing import List, Tuple

def weighted_mean(vals, weights):
    return (vals * weights).sum() / weights[~np.isnan(vals)].sum()

def normalize_with_mean_and_std(vals, weights):
    mean = weighted_mean(vals, weights)
    std = np.sqrt(weighted_mean((vals - mean)**2, weights))
    return (vals - mean) / std

def get_sorted(tracks_mi: pd.DataFrame, field: str | List[str], by: str) -> Tuple[pd.DataFrame | pd.Series, pd.Series]:
    
    idxs = np.argsort(tracks_mi[by])[::-1]
    field_sorted = tracks_mi[field].iloc[idxs]
    slots_sorted = tracks_mi['slots'].iloc[idxs]
    
    return field_sorted, slots_sorted

