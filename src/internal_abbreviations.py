import pandas as pd
from typing import List


def format_well_id(well):
    experiment = well['experiment']
    inhibitor_str = format_positive_inhibitors(well)
    exposition_str = format_number(well['exposition'])
    repetition = well['repetition']
    return f"{experiment}--{exposition_str}s-{inhibitor_str}-rep{repetition}"


def format_number(x):
    x = int(x) if x.is_integer() else x
    x_str = str(x).replace('.', '')
    return x_str


def format_positive_inhibitors(well_info, types_only=False, unit_separator='', inhibitors_separator='', no_inh='0uM'):
    inhibitor_cols = well_info[[col for col in well_info.index if col.startswith('inh_')]]
    inhibitors_positive_cols = inhibitor_cols[inhibitor_cols > 0]
    inhibitor_str = _format_inhibitor(inhibitors_positive_cols.index, inhibitors_positive_cols if not types_only else None, unit_separator=unit_separator, inhibitors_separator=inhibitors_separator, no_inh=no_inh)
    return inhibitor_str


def _format_inhibitor(inhibitor_cols, inhibitor_concentrations=None, unit_separator='', inhibitors_separator='', no_inh='0uM'):
    if not len(inhibitor_cols):
        return no_inh
    else:
        if inhibitor_concentrations is None:
            return inhibitors_separator.join(
                f'{inh_col[4:8]}'
                for inh_col in inhibitor_cols
            )
        return inhibitors_separator.join(
            f'{inh_col[4:8]}{unit_separator}{format_number(inhibitor_concentration)}uM'
            for inh_col, inhibitor_concentration in zip(inhibitor_cols, inhibitor_concentrations)
        )


def has_exactly_inhibitors(well_info: pd.Series, inhibitors: List[str] | str, inh_concentrations: List[str] | None = None):
    '''If inh_concentration is None (default), checks if concentrations
of given inhibitors are non-zero and other are zero.
Otherwise, checks if inhibitor concentrations in well_info match inh_concentrations.'''
    if isinstance(inhibitors, str):
        inhibitors = [inhibitors]
        if inh_concentrations is not None:
            inh_concentrations = [inh_concentrations]
    inhibitor_cols = [f"inh_{inh}" for inh in inhibitors]
    if inh_concentrations is None:
        return well_info.reindex(inhibitor_cols).all() and not well_info.reindex([col for col in well_info.index if col.startswith('inh_') and col not in inhibitor_cols]).any()
    return (well_info.reindex(inhibitor_cols) == inh_concentrations).all() and not well_info.reindex([col for col in well_info.index if col.startswith('inh_') and col not in inhibitor_cols]).any()
