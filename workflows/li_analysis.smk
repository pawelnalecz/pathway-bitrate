from pathlib import Path
import sys
sys.path.append(str(Path('.').resolve()))

import matplotlib
import numpy as np
import pandas as pd
from subplots_from_axsize import subplots_from_axsize

from jax import numpy as jnp
from jax.scipy.special import logsumexp


from src import jax_evaluation
from src import jax_protocol
from src import jax_nn

from generic._defaults import (
    DATA_MANAGER,
    WELLS_SELECTED,
    SET_TYPES,
    DATASET_IDS,
    MODEL_IDS,
    TRAIN_IDS,
    TEST_IDS,
)

from config.local_config import OUTPUT_PATH

matplotlib.use("agg")
workdir: OUTPUT_PATH

# CONFIGS

TRAIN_IDS = ['main-q0', 'main-q0+opt-5-90protocol-L1']
TEST_IDS = ['q1', 'q1-reweight-optprotocol']

TEST_SET_TYPES = [
    'main',
    'main+cell+inh',
]

PLOTS = [
    'nn-predictions-heatmap',
    'nn-predictions-diagonal',
    # 'bitrate-by-interval',
    ]

# INCLUDE CORE RULES

include: "generic/_core_rules.smk"
include: "generic/_combine_plots.smk"
include: "generic/_evaluate_network.smk"


# RULES

rule all:
    input:
        expand(
            'li_analysis/combined/per_experiment/{experiment}/{plot}_{dataset_id}_{model_id}_{train_id}_{test_id}.png',
            experiment=WELLS_SELECTED['experiment'].unique(),
            dataset_id=DATASET_IDS,
            model_id=MODEL_IDS,
            train_id=TRAIN_IDS,
            test_id=TEST_IDS,
            plot=PLOTS,
        ),
        expand(
            'li_analysis/combined/per_set_type/{set_type}/{plot}_{dataset_id}_{model_id}_{train_id}_{test_id}.png',
            set_type=TEST_SET_TYPES,
            dataset_id=DATASET_IDS,
            model_id=MODEL_IDS,
            train_id=TRAIN_IDS,
            test_id=TEST_IDS,
            plot=PLOTS,
        ),



rule plot_bitrate_by_interval:
    input:
        predictions='cache/predictions/{per}/{well_or_set_id}/predictions_{dataset_id}_{model_id}_{train_id}_{test_id}.csv.gz',
        protocol=inputs_test_protocol,
    output:
        'li_analysis/single/{per}/{well_or_set_id}/bitrate-by-interval_{dataset_id}_{model_id}_{train_id}_{test_id}.png'
    resources:
        mem_gib=1
    run:
        protocol = jax_protocol.Protocol(path=input.protocol)
        predictions = pd.read_csv(input.predictions)

        bph = protocol.dt / np.log(2)

        fig, axs = subplots_from_axsize(ncols=3, axsize=(4, 3), top=0.6, right=0.4)

        intervals = predictions['I'].unique()
        i_idxs = protocol.get_l_idx(intervals)
        batch_i_idxs = protocol.get_l_idx(jnp.array(predictions['I']))
        batch_l_idxs = protocol.get_l_idx(jnp.array(predictions['L']))

        if 'importance_weight' not in predictions:
            predictions['importance_weight'] = 1.
        mi_ce = jnp.array([
            jax_evaluation.mutual_information_cross_per_slot_from_predictions(predictions_part)
            / predictions_part['importance_weight'].mean() # necessary 
            for interval, predictions_part in predictions.groupby('I')
        ])

        reshaped_mi_ce = jnp.zeros_like(protocol.ls, dtype=jnp.float64).at[i_idxs].set(mi_ce)
        avg_mi_ce = jax_evaluation.mutual_information_cross_per_slot_from_predictions(predictions)

        mi_ce_bph = mi_ce * bph

        ax = axs[0]
        ax.plot(intervals / protocol.dt, mi_ce_bph, marker='o', ms=3)
        ax.set_ylim(bottom=min(0, mi_ce_bph.min() * 1.1))
        ax.annotate(f"mi_ce: {avg_mi_ce * bph:.2f}", (.6, .7), xycoords='axes fraction')




        next_interval_logprobs = jnp.array(protocol.interval_logprobs) + (reshaped_mi_ce - reshaped_mi_ce.mean()) / avg_mi_ce
        next_interval_logprobs -= logsumexp(next_interval_logprobs)
        next_protocol = protocol.update_interval_logprobs(next_interval_logprobs)

        li_logprobs = protocol.li_joint_log_probabilities()
        next_li_logprobs = next_protocol.li_joint_log_probabilities()
        li_logprobs_diff = jnp.nan_to_num(next_li_logprobs - li_logprobs, nan=0.)
        batch_li_probs_ratio = jnp.exp(li_logprobs_diff)[batch_l_idxs, batch_i_idxs]

        axs[1].plot(next_protocol.ls / next_protocol.dt, jnp.exp(next_protocol.interval_logprobs))

        next_stopping_logprobs = next_protocol.stopping_log_probabilities()
        next_stopping_logits = jax_nn.logprob_to_logit(next_stopping_logprobs)
        next_batch_stop_logits = next_stopping_logits[batch_l_idxs]

        next_batch_predicted_p_logits = jnp.array(predictions['p_network_logit']) + next_batch_stop_logits

        next_predictions = pd.DataFrame({
            'L': predictions['L'],
            'I': predictions['I'],
            'p_prior_logit': next_batch_stop_logits,
            'p_network_logit': predictions['p_network_logit'],
            'p_predicted_logit': next_batch_predicted_p_logits,
            'y': predictions['y'],
            'importance_weight': predictions['importance_weight'] * batch_li_probs_ratio,
        })

        mi_ce = jnp.array([
            jax_evaluation.mutual_information_cross_per_slot_from_predictions(predictions_part)
            / predictions_part['importance_weight'].mean() # necessary 
            for interval, predictions_part in next_predictions.groupby('I')
        ])
        avg_mi_ce = jax_evaluation.mutual_information_cross_per_slot_from_predictions(next_predictions)

        mi_ce_bph = mi_ce * protocol.dt / np.log(2)


        ax = axs[2]
        ax.plot(intervals / protocol.dt, mi_ce_bph, marker='o', ms=3)
        ax.set_ylim(bottom=min(0, mi_ce_bph.min() * 1.1))
        ax.annotate(f"mi_ce: {avg_mi_ce * bph:.2f}", (.6, .7), xycoords='axes fraction')

        fig.suptitle(f'{wildcards.well_or_set_id} -{wildcards.test_id}\n{wildcards.dataset_id}; {wildcards.model_id}; {wildcards.train_id};')
        fig.savefig(str(output), dpi=300)


