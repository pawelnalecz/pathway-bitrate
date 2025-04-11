import matplotlib
import pandas as pd
import numpy as np
from subplots_from_axsize import subplots_from_axsize

from generic._defaults import (
    DATA_MANAGER,
    WELLS_SELECTED,
    SET_TYPES,
    TEST_IDS,
)

from config.configs import (
    TEST_CONFIGS,
)

from src import jax_plots
from config.local_config import OUTPUT_PATH

matplotlib.use("agg")
workdir: OUTPUT_PATH

# CONFIGS

TEST_SET_TYPES = ['main+cell+inh']
QUALITIES = ['q0', 'q1']

# INCLUDE CORE RULES

include: "generic/_combine_plots.smk"
include: "generic/_core_rules.smk"
include: "generic/_measures.smk"


# RULES

rule all:
    input:
        expand(
            'plot_traditional_analysis/combined/per_experiment/{experiment}/{plot_type}_{quality}.png',
            plot_type=[
                'response-amplitude-by-interval',
                'response-amplitude-over-reference-by-interval',
            ],
            experiment=WELLS_SELECTED['experiment'].unique(),
            quality=QUALITIES,
        ),
        expand(
            'plot_traditional_analysis/combined/per_set_type/{set_type}/{plot_type}_{quality}.png',
            plot_type=[
                'response-amplitude-by-interval',
                'response-amplitude-over-reference-by-interval',
            ],
            set_type=TEST_SET_TYPES,
            quality=QUALITIES,
        )

rule plot_response_amplitude_by_interval:
    input:
        tracks_info=inputs_tracks_info,
        tracks_preprocessed=inputs_tracks_preprocessed,
    output:
        'plot_traditional_analysis/single/{per}/{well_or_set_id}/response-amplitude-by-interval_q{quality}.png'
    resources:
        mem_gib=1
    run:
        tracks_preprocessed, seconds_per_timepoint = get_tracks_preprocessed_with_quality(wildcards, input)

        fig, ax = subplots_from_axsize(axsize=(6,3), top=.5)

        jax_plots.plot_response_amplitude_by_interval(ax, tracks_preprocessed, seconds_per_timepoint=seconds_per_timepoint)
        
        ax.set_title(f'{wildcards.well_or_set_id}')
        fig.savefig(str(output), dpi=300)



rule plot_response_amplitude_over_reference_by_interval:
    input:
        tracks_info=inputs_tracks_info,
        tracks_preprocessed=inputs_tracks_preprocessed,
    output:
        'plot_traditional_analysis/single/{per}/{well_or_set_id}/response-amplitude-over-reference-by-interval_q{quality}.png'
    resources:
        mem_gib=1
    run:
        tracks_preprocessed, seconds_per_timepoint = get_tracks_preprocessed_with_quality(wildcards, input)

        fig, ax = subplots_from_axsize(axsize=(6,3), top=.5)

        jax_plots.plot_log_response_over_reference_by_interval(ax, tracks_preprocessed, seconds_per_timepoint=seconds_per_timepoint)
        
        ax.set_title(f'{wildcards.well_or_set_id}')
        fig.savefig(str(output), dpi=300)
