import pandas as pd

from scipy.special import expit, log_expit

import jax.numpy as jnp

from src.jax_nn import NnModel
from src.jax_dataloader import DataLoader
from src.jax_protocol import Protocol


def evaluate_on_full_dataset(
    model: NnModel,
    dataloader: DataLoader,
    protocol: Protocol,
    extra_cols=['track_id', 'time_in_seconds'],
    reweight=False,  # TODO(frdrc): currently unused
):
    if reweight:
        batch = dataloader.get_full_batch_for_protocol(protocol, extra_cols=extra_cols)
    else:
        batch = dataloader.get_full_batch(protocol, extra_cols=extra_cols)

    # check that batch and protocol L indexes match
    assert batch['dt'] == protocol.dt
    assert batch['l_idxs'].max() < len(protocol.ls)
        
    batch_update_logits = model.compute_update_logits(batch)
    batch_prior_logits = model.compute_prior_logits(protocol, batch)
    batch_posterior_logits = batch_update_logits + batch_prior_logits
    
    predictions = pd.DataFrame({
        'L': protocol.ls[batch['l_idxs']],
        'I': protocol.ls[batch['i_idxs']],
        'p_prior_logit': batch_prior_logits,
        'p_network_logit': batch_update_logits,
        'p_predicted_logit': batch_posterior_logits,
        'y': batch['ys'],
    })

    for col in extra_cols:
        predictions[col] = batch[col]

    if reweight:
        # compute expectations with (f * importance_weight).mean()
        # remember that f is assumed to be zero for (L, I) not in the dataset!
        li_logprobs = protocol.li_joint_log_probabilities()
        batch_log_ws = li_logprobs[batch['l_idxs'], batch['i_idxs']] - batch['sampling_li_logprobs']
        batch_ws = jnp.exp(batch_log_ws)
        predictions['importance_weight'] = batch_ws

    return predictions



def naive_entropy_from_logit(p_logit):
    return -(expit(p_logit) * log_expit(p_logit) + expit(-p_logit) * log_expit(-p_logit))


def cross_entropy_from_logit(p_logit, y):
    return -(log_expit(p_logit) * y + log_expit(-p_logit) * (1 - y))


def reconstruction_naive_entropy_per_slot_from_predictions(predictions) -> float:
    ws = predictions.get('importance_weight', 1.0)
    re_naive = naive_entropy_from_logit(predictions['p_predicted_logit'])
    return (re_naive * ws).mean()


def reconstruction_cross_entropy_per_slot_from_predictions(predictions) -> float:
    ws = predictions.get('importance_weight', 1.0)
    re_cross = cross_entropy_from_logit(predictions['p_predicted_logit'], predictions['y'])
    return (re_cross * ws).mean()


def mutual_information_naive_per_slot_from_predictions(predictions) -> float:
    ws = predictions.get('importance_weight', 1.0)
    re_naive_prior = naive_entropy_from_logit(predictions['p_prior_logit'])
    re_naive = naive_entropy_from_logit(predictions['p_predicted_logit'])
    entropy_gain = re_naive_prior - re_naive
    return (entropy_gain * ws).mean()


def mutual_information_cross_per_slot_from_predictions(predictions) -> float:
    ws = predictions.get('importance_weight', 1.0)
    re_cross_prior = cross_entropy_from_logit(predictions['p_prior_logit'], predictions['y'])
    re_cross = cross_entropy_from_logit(predictions['p_predicted_logit'], predictions['y'])
    entropy_gain = re_cross_prior - re_cross
    return (entropy_gain * ws).mean()
