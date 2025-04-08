import re
import yaml

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp, xlogy
from jax.scipy.stats import gamma


def li_joint_log_probabilities(interval_logprobs):
    """Assumes len(interval_logprobs) == len(ls)!"""
    n = len(interval_logprobs)
    li_logprobs = jnp.log(jnp.triu(jnp.ones((n, n))))
    li_logprobs = li_logprobs + interval_logprobs[None, :]
    li_logprobs -= jax.scipy.special.logsumexp(li_logprobs)
    return li_logprobs


def stopping_log_probabilities(interval_logprobs):
    """Assumes len(interval_logprobs) == len(ls)!"""
    remaining_logprobs = jax.lax.cumlogsumexp(interval_logprobs[::-1])[::-1]
    stop_logprobs = interval_logprobs - remaining_logprobs
    return stop_logprobs


def entropy_per_slot(interval_logprobs):
    """Assumes len(interval_logprobs) == len(ls)!"""
    interval_probs = jnp.exp(interval_logprobs)
    entropy_per_interval = -xlogy(interval_probs, interval_probs).sum()
    avg_interval_length = (interval_probs * jnp.arange(1, 1+len(interval_probs))).sum()
    entropy_per_slot = entropy_per_interval / avg_interval_length
    return entropy_per_slot


class Protocol:
    def __init__(self, interval_logprobs=None, name="Unnamed protocol", dt=60, path=None):
        if path is not None:
            with open(path) as f:
                protocol_dict = yaml.safe_load(f)
            self.name = protocol_dict['name']
            self.dt = protocol_dict['dt']
            self.interval_logprobs = jnp.array(protocol_dict['interval_logprobs'])
        else:
            self.name = name
            self.dt = dt
            interval_logprobs -= logsumexp(interval_logprobs)
            self.interval_logprobs = jnp.array(interval_logprobs)

        self.slots = jnp.arange(1, 1 + self.interval_logprobs.shape[0])
        self.ls = dt * self.slots
        
    def save(self, path):
        with open(path, 'w') as fout:
            yaml.dump({
                'name': self.name,
                'dt': self.dt,
                'interval_logprobs': self.interval_logprobs.tolist(),
            }, 
            stream=fout,
            sort_keys=False,
            )

    def update_interval_logprobs(self, interval_logprobs):
        return Protocol(interval_logprobs, name=self.name, dt=self.dt)

    def li_joint_log_probabilities(self) -> jax.Array:
        return li_joint_log_probabilities(self.interval_logprobs)

    def stopping_log_probabilities(self):
        return stopping_log_probabilities(self.interval_logprobs)

    def entropy_per_slot(self):
        return entropy_per_slot(self.interval_logprobs)

    def get_l_idx(self, ls):
        return jnp.searchsorted(self.ls, ls)

def parse_intervals(match, min_group=1, max_group=2):
    min_interval = 60 * int(match.group(min_group))
    max_interval = 60 * int(match.group(max_group))
    intervals = jnp.arange(min_interval, max_interval + 1, 60)  
    return intervals


def first_zero_to_decimal(num_str):
    if num_str.startswith('0'):
        num_str = '0.' + num_str[1:]
    return float(num_str)


# TODO(frdrc): for compatibility with new dataset, fix later
def create_protocol_from_intervals(
    intervals,
    interval_logprobs,
    dt=60,
    **kwargs,
):
    ls = jnp.arange(intervals.max(), 0, -dt)[::-1]
    interval_idxs = jnp.searchsorted(ls, intervals)
    interval_logprobs_like_ls = jnp.full(len(ls), -jnp.inf)
    interval_logprobs_like_ls = interval_logprobs_like_ls.at[interval_idxs].set(interval_logprobs)
    return Protocol(interval_logprobs_like_ls, dt=dt, **kwargs)


def create_named_protocol(protocol_type) -> Protocol:
    exp_protocol_match = re.fullmatch(r"([0-9]+)\-([0-9]+)exp([0-9]+)", protocol_type)
    gamma_protocol_match = re.fullmatch(r"([0-9]+)\-([0-9]+)gamma([0-9]+)x([0-9]+)", protocol_type)
    uniform_protocol_match = re.fullmatch(r"([0-9]+)\-([0-9]+)uniform", protocol_type)
    if exp_protocol_match is not None:
        intervals = parse_intervals(exp_protocol_match)
        alpha = -first_zero_to_decimal(exp_protocol_match.group(3))
        return geometric_protocol(
            alpha=alpha,
            intervals=intervals,
            name=protocol_type,
        )
    elif gamma_protocol_match is not None:
        intervals = parse_intervals(gamma_protocol_match)
        n = first_zero_to_decimal(gamma_protocol_match.group(3))
        scale = first_zero_to_decimal(gamma_protocol_match.group(4))
        return gamma_protocol(
            n=n,
            scale=scale,
            intervals=intervals,
            name=protocol_type,
        )
    elif uniform_protocol_match is not None:
        intervals = parse_intervals(uniform_protocol_match)
        return uniform_protocol(
            intervals=intervals,
            name=protocol_type,
        )

    else:
        assert False, 'Unknown protocol!'


def geometric_protocol(alpha, intervals, **kwargs):
    interval_logprobs = alpha * jnp.arange(len(intervals), dtype=jnp.float32)
    return create_protocol_from_intervals(intervals, interval_logprobs, **kwargs)


def gamma_protocol(n, scale, intervals, **kwargs):
    intervals_jnp = jnp.array(intervals, dtype=jnp.float32) / 60
    interval_probs = gamma.pdf(intervals_jnp + 1e-5, n, scale=scale)
    interval_logprobs = jnp.log(interval_probs / interval_probs.sum())
    return create_protocol_from_intervals(intervals, interval_logprobs, **kwargs)


def uniform_protocol(intervals, **kwargs):
    interval_logprobs = jnp.ones_like(intervals) * (1 / len(intervals))
    return create_protocol_from_intervals(intervals, interval_logprobs, **kwargs)

