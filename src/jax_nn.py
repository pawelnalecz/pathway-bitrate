from typing import Optional, Sequence
from datetime import datetime
import json

import pandas as pd
import numpy as np

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from src import jax_utils
from src import jax_protocol
from src.jax_dataloader import DataLoader
from src.jax_protocol import Protocol


class MLP(eqx.Module):
    layers: list
    means: jax.Array
    stds: jax.Array

    def __init__(
        self,
        key: jax.Array,
        dim_input: int,
        hidden_layers: Sequence[int],
        means: Optional[jax.Array] = None,
        stds: Optional[jax.Array] = None,
    ) -> None:
        key_final, key_hidden = jax.random.split(key)
        keys_hidden = jax.random.split(key_hidden, len(hidden_layers))

        dims = [dim_input] + list(hidden_layers)

        self.layers = []
        for dim_in, dim_out, key in zip(dims, dims[1:], keys_hidden):
            self.layers.append(eqx.nn.Linear(dim_in, dim_out, key=key))

        self.layers.append(eqx.nn.Linear(dims[-1], 1, key=key_final))
        self.means = jnp.zeros((dim_input,)) if means is None else means
        self.stds = jnp.ones((dim_input,)) if stds is None else stds

    def call(self, x: jax.Array) -> jax.Array:
        z = jax.lax.stop_gradient((x - self.means) / (self.stds + 1e-6))

        for layer in self.layers[:-1]:
            z = layer(z)
            z = jax.nn.leaky_relu(z)

        # last layer without activation
        z = self.layers[-1](z)

        return z[..., 0]  # return scalar


class NnModel:
    def __init__(
        self,
        dataloader: Optional[DataLoader] = None,
        path: Optional[str] = None,
        hidden_layers: Sequence = (20, 10),
        normalize: bool = True,
        seed: int = 0,
    ):
        self.dataloader = dataloader
        self.path = path
        self.hidden_layers = tuple(hidden_layers)
        self.normalize = normalize
        self.protocol = None
        self.seed = seed

        if path is not None:
            self.load(path)
        
        elif self.dataloader is not None:
            self.dim_input = self.dataloader.dim_xs
            self._init_mlp()
        
        else:
            raise ValueError('Either dataloader or load path should be provided!')

        self.training_log = pd.DataFrame()

    def __str__(self):
        return f'NN ' + str(self.hidden_layers)

    def _init_mlp(self):
        assert self.dataloader is not None
        key = jax.random.key(self.seed)
        self.mlp = MLP(
            key,
            dim_input=self.dim_input,
            hidden_layers=self.hidden_layers,
            means=self.dataloader.dataset_xs.mean(axis=0) if self.normalize else None,
            stds=self.dataloader.dataset_xs.std(axis=0) if self.normalize else None,
        )

    ## input-output
    def save(self, path):
        with open(path, 'wb') as f:
            hyperparams_str = json.dumps({
                'dim_input': self.dim_input,
                'hidden_layers': self.hidden_layers,
            })
            f.write((hyperparams_str + "\n").encode())
            eqx.tree_serialise_leaves(f, self.mlp)

    def load(self, path):
        with open(path, 'rb') as f:
            hyperparams = json.loads(f.readline().decode())
            self.dim_input = hyperparams['dim_input']
            self.hidden_layers = hyperparams['hidden_layers']
            self.mlp = MLP(jax.random.PRNGKey(0), self.dim_input, self.hidden_layers)
            self.mlp = eqx.tree_deserialise_leaves(f, self.mlp)

    ## evaluation
    def __call__(self, protocol: Protocol, batch):
        logits = self.compute_posterior_logits(protocol, batch)
        probs = jax.nn.sigmoid(logits)
        return probs

    def compute_posterior_logits(self, protocol, batch):
        # check that batch and protocol L indexes match
        assert batch['dt'] == protocol.dt
        assert batch['l_idxs'].max() < len(protocol.ls)
        return _compute_posterior_logits(self.mlp, protocol.interval_logprobs, batch)

    def compute_update_logits(self, batch):
        return _compute_update_logits(self.mlp, batch)

    def compute_prior_logits(self, protocol, batch):
        assert batch['dt'] == protocol.dt
        assert batch['l_idxs'].max() < len(protocol.ls)
        return _compute_prior_logits(protocol.interval_logprobs, batch)

    ## training
    def train(
        self,
        protocol: Optional[Protocol],
        dataloader: Optional[DataLoader], # if provided, overrides self.dataloader
        train_steps=5_000,
        batch_size=10_000,
        lr=1e-2,
        weight_decay=1e-4,
        optimize_network=True,
        optimize_protocol=False,
        protocol_d2_L1_penalty=0.,
        protocol_d2_L2_penalty=0.,
        log_every=10,
        verbose=False,
        seed=0,
    ):
        dataloader = dataloader or self.dataloader
        assert dataloader is not None
        assert dataloader.dim_xs == self.dim_input
        assert optimize_network or optimize_protocol
        self.dataloader = dataloader
        if protocol is None:
            assert self.protocol is not None
            protocol = self.protocol

        key = jax.random.key(seed)
        
        # create batching and training step functions
        get_batch_fn = self._create_get_batch_fn(
            protocol=protocol,
        )

        step_fn = self._create_step_fn(
            optimize_network=optimize_network,
            optimize_protocol=optimize_protocol,
            protocol_d2_L1_penalty=protocol_d2_L1_penalty,
            protocol_d2_L2_penalty=protocol_d2_L2_penalty,
            interval_logprobs_mask=jnp.isfinite(protocol.interval_logprobs),
        )

        mlp = self.mlp

        # init optimizer, mask out weight decay on normalization constants
        mask_network = jax.tree.map(lambda _: True, mlp)
        mask_network = eqx.tree_at(
            lambda tree: (tree.means, tree.stds),
            mask_network,
            replace=(False, False)
        )
        mask = (
            (mask_network, False) if optimize_protocol and optimize_network
            else mask_network if optimize_network
            else False
        )
        params = (
            (mlp, protocol.interval_logprobs) if optimize_protocol and optimize_network
            else mlp if optimize_network
            else protocol.interval_logprobs
        )
        optim = optax.adamw(lr, weight_decay=weight_decay, mask=mask)
        opt_state = optim.init(params)

        # training loop
        training_log = []
        start = self.training_log['step'].max() + 1 if len(self.training_log) else 1
        for i, key_batch in enumerate(jax.random.split(key, train_steps), start=start):
            # get batch
            batch = get_batch_fn(
                key_batch,
                protocol.interval_logprobs,
                batch_size=batch_size,
            )
            
            # compute step
            pe = protocol.entropy_per_slot()
            mlp, interval_logprobs_new, opt_state, info = step_fn(
                mlp,
                protocol.interval_logprobs,
                batch,
                opt_state,
                optim,
            )
            protocol = protocol.update_interval_logprobs(interval_logprobs_new)

            # logging
            if i % log_every == 0:
                cmi = info['cmi']
                cre = pe - cmi
                pe_bph = jax_utils.nat_per_slot_to_bit_per_hour(pe)
                cre_bph = jax_utils.nat_per_slot_to_bit_per_hour(cre)
                cmi_bph = jax_utils.nat_per_slot_to_bit_per_hour(cmi)

                entry = {
                    'step': i,
                    'timestamp': datetime.now(),
                    'PE': float(pe_bph),
                    'cRE': float(cre_bph),
                    'cMI': float(cmi_bph),
                    'loss': float(info['loss']),
                    'loss_d2_L1': float(info['loss_d2_L1']),
                    'loss_d2_L2': float(info['loss_d2_L2']),
                }

                if optimize_protocol:
                    entry['protocol'] = protocol.interval_logprobs.tolist()

                training_log.append(entry)

                if verbose:
                    print(
                        f'step: {i:4d}\t'
                        f' PE: {pe_bph:.2f} [bit/hour]\t'
                        f'cRE: {cre_bph:.2f} [bit/hour]\t'
                        f'cMI: {cmi_bph:.2f} [bit/hour]'
                    )
                    
        # store results
        self.mlp = mlp
        self.training_log = pd.concat([self.training_log, pd.DataFrame(training_log)])
        self.protocol = protocol

    def _create_get_batch_fn(self, protocol, jit=True):
        def get_batch_fn(key, interval_logprobs, batch_size):
            protocol_current = protocol.update_interval_logprobs(interval_logprobs)
            batch = self.dataloader.get_batch(key, protocol_current, batch_size=batch_size)
            return batch

        if jit:
            get_batch_fn = jax.jit(get_batch_fn, static_argnames=['batch_size'])

        return get_batch_fn

    def _create_step_fn(
        self,
        optimize_network,
        optimize_protocol,
        protocol_d2_L1_penalty,
        protocol_d2_L2_penalty,
        interval_logprobs_mask,
        jit=True,
    ):
        interval_logprobs_mask = np.array(interval_logprobs_mask)

        def loss_fn(mlp, interval_logprobs, batch):
            minus_cmi = -_compute_cross_mutual_information_per_slot(mlp, interval_logprobs, batch)
            interval_logprobs_optimized = interval_logprobs[interval_logprobs_mask]
            loss_d2_L1 = protocol_d2_L1_penalty * jnp.abs(jnp.diff(interval_logprobs_optimized, n=2)).mean()
            loss_d2_L2 = protocol_d2_L2_penalty * jnp.square(jnp.diff(interval_logprobs_optimized, n=2)).mean()
            loss = minus_cmi + loss_d2_L1 + loss_d2_L2
            return loss, dict(
                cmi=-minus_cmi,
                loss_d2_L1=loss_d2_L1,
                loss_d2_L2=loss_d2_L2,
                loss=loss,
            )

        if optimize_network and optimize_protocol:
            def step_fn(
                mlp,
                interval_logprobs,
                batch,
                opt_state,
                optim,
            ):
                (grad_mlp, grad_interval_logprobs), info = jax.grad(loss_fn, has_aux=True, argnums=(0, 1))(
                    mlp,
                    interval_logprobs,
                    batch,
                )

                # mask out gradient (usually at zero probabilities)
                grad_interval_logprobs = jnp.where(interval_logprobs_mask, grad_interval_logprobs, 0.0)

                # apply "sum of probabilities = 1" constraint to interval_logprobs gradient
                constraint_tangent = jnp.exp(interval_logprobs)
                grad_interval_logprobs = _make_orthogonal_to(
                    grad_interval_logprobs,
                    constraint_tangent,
                )

                # apply gradient update
                grad = (grad_mlp, grad_interval_logprobs)
                updates, opt_state = optim.update(grad, opt_state, (mlp, interval_logprobs))
                mlp, interval_logprobs = eqx.apply_updates((mlp, interval_logprobs), updates)

                # renormalize interval_logprobs
                interval_logprobs -= jax.scipy.special.logsumexp(interval_logprobs)

                return mlp, interval_logprobs, opt_state, info

        elif optimize_protocol:
            def step_fn(
                mlp,
                interval_logprobs,
                batch,
                opt_state,
                optim,
            ):
                grad_interval_logprobs, info = jax.grad(loss_fn, has_aux=True, argnums=1)(
                    mlp,
                    interval_logprobs,
                    batch,
                )

                # mask out gradient (usually at zero probabilities)
                grad_interval_logprobs = jnp.where(interval_logprobs_mask, grad_interval_logprobs, 0.0)

                # apply "sum of probabilities = 1" constraint to interval_logprobs gradient
                constraint_tangent = jnp.exp(interval_logprobs)
                grad_interval_logprobs = _make_orthogonal_to(
                    grad_interval_logprobs,
                    constraint_tangent,
                )

                # apply gradient update
                updates, opt_state = optim.update(grad_interval_logprobs, opt_state, interval_logprobs)
                interval_logprobs = eqx.apply_updates(interval_logprobs, updates)

                # renormalize interval_logprobs
                interval_logprobs -= jax.scipy.special.logsumexp(interval_logprobs)

                return mlp, interval_logprobs, opt_state, info

        elif optimize_network:
            def step_fn(
                mlp,
                interval_logprobs,
                batch,
                opt_state,
                optim,
            ):
                grad, info = jax.grad(loss_fn, has_aux=True)(
                    mlp,
                    interval_logprobs,
                    batch,
                )
                
                updates, opt_state = optim.update(grad, opt_state, mlp)
                mlp = eqx.apply_updates(mlp, updates)
                
                return mlp, interval_logprobs, opt_state, info

        else:
            raise ValueError('Either optimize_network or optimize_protocol must be True')

        if jit:
            step_fn = jax.jit(step_fn, static_argnames=['optim'])

        return step_fn


def _protocol_stopping_logits(interval_logprobs, clip_limit=1e6):
    interval_logprobs = jnp.clip(interval_logprobs, a_min=-clip_limit)
    remaining_logprobs = jax.lax.cumlogsumexp(interval_logprobs[::-1])[::-1]
    stop_logits = interval_logprobs - jnp.roll(remaining_logprobs, -1)
    stop_logits = stop_logits.at[-1].set(clip_limit)
    return stop_logits


def _compute_prior_logits(interval_logprobs, batch):
    stopping_logits = _protocol_stopping_logits(interval_logprobs)
    batch_prior_logits = stopping_logits[batch['l_idxs']]
    return batch_prior_logits


def _compute_update_logits(mlp, batch):
    """Compute logit update to prior."""
    return jax.vmap(mlp.call)(batch['xs'])


def _compute_posterior_logits(mlp, interval_logprobs, batch):
    batch_prior_logits = _compute_prior_logits(interval_logprobs, batch)
    batch_update_logits = _compute_update_logits(mlp, batch)
    batch_posterior_logits = batch_update_logits + batch_prior_logits
    return batch_posterior_logits


def _compute_cross_mutual_information_per_slot(mlp, interval_logprobs, batch):
    # compute prior/posterior predictions
    batch_prior_logits = _compute_prior_logits(interval_logprobs, batch)
    batch_update_logits = _compute_update_logits(mlp, batch)
    batch_posterior_logits = batch_update_logits + batch_prior_logits

    # compute importance weights
    li_logprobs = jax_protocol.li_joint_log_probabilities(interval_logprobs)
    batch_log_weights = li_logprobs[batch['l_idxs'], batch['i_idxs']] - batch['sampling_li_logprobs']
    batch_weights = jnp.exp(batch_log_weights)

    # compute average change in surprisal
    batch_prior_surprisals = optax.sigmoid_binary_cross_entropy(batch_prior_logits, batch['ys'])
    batch_posterior_surprisals = optax.sigmoid_binary_cross_entropy(batch_posterior_logits, batch['ys'])
    mi = (batch_weights * (batch_prior_surprisals - batch_posterior_surprisals)).mean()

    return mi


def _make_orthogonal_to(vs, cs):
    return vs - cs * (vs * cs).sum() / (cs * cs).sum()
