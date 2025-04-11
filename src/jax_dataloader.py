import pandas as pd

import jax
import jax.numpy as jnp

from src.jax_utils import jax_assert
from src.jax_protocol import Protocol
from config.parameters import KEEP_START, KEEP_END

class DataLoader:
    def __init__(
        self,
        dataset,
        input_cols=None,
        impute=True,
        normalize=False,  # TODO(frdrc): unused
    ):
        assert len(dataset), "Dataset empty!"
        self.dataset = dataset
        self.impute = impute

        # TODO(frdrc): remove X_log_l from dataset CSV

        # useful
        self.ls = jnp.sort(jnp.array(self.dataset['L'].unique()))
        self.intervals = jnp.sort(jnp.array(self.dataset['I'].unique()))

        # sort and prepare slice ranges for sampling
        self.dataset['LI_hash'] = _cantor_pairing(self.dataset['L'], self.dataset['I'])
        self.dataset = self.dataset.sort_values(by=['LI_hash'])
        self.dataset = self.dataset.reset_index(drop=True)
        self.dataset = self.dataset.reset_index(drop=False)
        li_slices = pd.DataFrame()
        li_slices['min'] = self.dataset.groupby(['LI_hash'])['index'].min()
        li_slices['max'] = self.dataset.groupby(['LI_hash'])['index'].max() + 1
        self.lookup_li_hash = jnp.array(li_slices.index)
        self.lookup_li_hash_min = jnp.array(li_slices['min'])
        self.lookup_li_hash_max = jnp.array(li_slices['max'])

        # prepare inputs
        if input_cols is None:
            input_cols = [col for col in self.dataset if col.startswith('X')]
            
        self.dataset_xs = jnp.array(self.dataset[input_cols])        
        self.dim_xs = len(input_cols)
        self.x_log_l_col = input_cols.index('X_log_l') if 'X_log_l' in input_cols else None
        
        if normalize:
            self.dataset_xs -= self.dataset_xs.mean(axis=0, keepdims=True)
            self.dataset_xs /= self.dataset_xs.std(axis=0, keepdims=True)

        # prepare correct predictions
        self.dataset_ys = jnp.array(self.dataset['L'] == self.dataset['I']).astype(int)

    def __len__(self):
        return len(self.dataset)

    def _impute_li(self, l, i):
        if self.impute:
            return _impute_li(l, i, self.intervals, self.ls)
        else:
            return l, i

    def _get_li_hash_idx(self, l, i):
        li_hash = _cantor_pairing(l, i)
        li_hash_idx = jnp.searchsorted(self.lookup_li_hash, li_hash)

        # set failed lookup to -1
        li_hash_idx = jnp.where(
            li_hash == self.lookup_li_hash[li_hash_idx],
            li_hash_idx,
            -1,
        )

        return li_hash_idx

    def check_protocol_compatible(self, protocol: Protocol):
        any_imputed = False
        for l_idx, l in enumerate(protocol.ls):
            for i_idx, i in enumerate(protocol.ls):
                if l > i or jnp.isinf(protocol.interval_logprobs[i_idx]):
                    continue

                l_imputed, i_imputed = self._impute_li(l, i)
                li_hash_idx = self._get_li_hash_idx(l_imputed, i_imputed)

                imputed = l_imputed != l or i_imputed != i
                any_imputed = any_imputed or imputed
                # if imputed:
                #     print(f"Imputing: (L={l / 60}, I={i / 60}) => (L={l_imputed / 60}, I={i_imputed / 60})!")

                assert l_imputed > 0, (
                    f"Imputation failed for: (L={l / 60}, I={i / 60}) => (L=?, I={i_imputed / 60})!"
                )

                assert li_hash_idx >= 0, (
                    f"Not in dataset: L = {l_imputed / 60}, I = {i_imputed / 60}"
                )
        if any_imputed:
            print(f"WARNING: Using data imputation!")


    def get_batch(self, key, protocol: Protocol, batch_size=1_000):
        key_li, key_slice = jax.random.split(key)

        li_logprobs = protocol.li_joint_log_probabilities()

        # sample L, I pairs from protocol
        li_probs_ravel = jnp.exp(li_logprobs.ravel())
        batch_li_ravel_idxs = jax.random.choice(
            key_li,
            len(li_probs_ravel),
            shape=(batch_size,),
            p=li_probs_ravel,
        )
        batch_l_idxs, batch_i_idxs = jnp.unravel_index(batch_li_ravel_idxs, li_logprobs.shape)
        batch_ls = protocol.ls[batch_l_idxs]
        batch_is = protocol.ls[batch_i_idxs]

        # impute to L, I available in dataset
        batch_ls_imputed, batch_is_imputed = jax.vmap(self._impute_li)(batch_ls, batch_is)
        jax_assert((batch_ls_imputed > 0).all(), "Imputation failed!")

        # find corresponding slices in dataset
        batch_li_hash_idxs = jax.vmap(self._get_li_hash_idx)(batch_ls_imputed, batch_is_imputed)
        jax_assert((batch_li_hash_idxs >= 0).all(), "Hash lookup failed!")
        batch_li_slice_mins = self.lookup_li_hash_min[batch_li_hash_idxs]
        batch_li_slice_maxs = self.lookup_li_hash_max[batch_li_hash_idxs]

        # sample according to slices
        batch_idxs = jax.random.randint(
            key_slice,
            shape=(batch_size,),
            minval=batch_li_slice_mins,
            maxval=batch_li_slice_maxs,
        )

        # insert correct X_log_l
        xs = self.dataset_xs[batch_idxs]
        if self.x_log_l_col is not None:
            xs = xs.at[:, self.x_log_l_col].set(jnp.log(batch_ls))
        
        batch = {
            'xs': xs,
            'ys': self.dataset_ys[batch_idxs],
            'i_idxs': batch_i_idxs,
            'l_idxs': batch_l_idxs,
            'sampling_li_logprobs': li_logprobs[batch_l_idxs, batch_i_idxs],
            'dt': protocol.dt,
        }

        return batch

    def get_full_batch(self, protocol, extra_cols=[]):
        """Return whole dataset as is. Needs protocol to compute L/I indices."""
        dataset_l_idxs = jnp.searchsorted(protocol.ls, jnp.array(self.dataset['L']))
        dataset_i_idxs = jnp.searchsorted(protocol.ls, jnp.array(self.dataset['I']))
        li_counts = jnp.array(self.dataset.groupby(['L', 'I']).transform('size'))
        return {
            'xs': self.dataset_xs,
            'ys': self.dataset_ys,
            'i_idxs': dataset_i_idxs,
            'l_idxs': dataset_l_idxs,
            'sampling_li_logprobs': jnp.log(li_counts) - jnp.log(len(li_counts)),
            'dt': protocol.dt,
        } | {
            col: jnp.array(self.dataset[col]) for col in extra_cols
        }

    def get_full_batch_for_protocol(self, protocol, extra_cols=['imputed']):
        """
        For each L, I in the protocol, collect all matching datapoints.
        Add sampling probability for computing importance weights down the line.
        """

        self.check_protocol_compatible(protocol)

        batch = {
            'xs': [],
            'ys': [],
            'i_idxs': [],
            'l_idxs': [],
            'sampling_li_logprobs': [],
        } | {col: [] for col in extra_cols}

        for l_idx, l in enumerate(protocol.ls):
            for i_idx, i in enumerate(protocol.ls):
                if l > i or jnp.isinf(protocol.interval_logprobs[i_idx]):
                    continue

                l_imputed, i_imputed = self._impute_li(l, i)
                li_hash_idx = self._get_li_hash_idx(l_imputed, i_imputed)
                li_slice_min = int(self.lookup_li_hash_min[li_hash_idx])
                li_slice_max = int(self.lookup_li_hash_max[li_hash_idx])

                li_count = li_slice_max - li_slice_min

                # insert correct X_log_l
                xs = self.dataset_xs[li_slice_min:li_slice_max]
                if self.x_log_l_col is not None:
                    xs = xs.at[:, self.x_log_l_col].set(jnp.log(l))

                batch['xs'].append(xs)
                batch['ys'].append(self.dataset_ys[li_slice_min:li_slice_max])
                batch['i_idxs'].append(jnp.full(li_count, i_idx))
                batch['l_idxs'].append(jnp.full(li_count, l_idx))
                batch['sampling_li_logprobs'].append(jnp.full(li_count, jnp.log(li_count)))

                if 'imputed' in extra_cols:
                    imputed = l_imputed != l or i_imputed != i
                    batch['imputed'].append(jnp.full(li_count, imputed))

                for col in extra_cols:
                    if col == 'imputed':
                        continue
                    batch[col].append(jnp.array(self.dataset.iloc[li_slice_min:li_slice_max][col]))

        for key, values in batch.items():
            batch[key] = jnp.concatenate(values)

        batch['sampling_li_logprobs'] -= jnp.log(len(batch['sampling_li_logprobs']))
        batch['dt'] = protocol.dt

        return batch


def _cantor_pairing(a, b):
    return (a + b) * (a + b + 1) // 2 + b


def _impute_l(l, i, i_target, keep_start=KEEP_START*60, keep_end=KEEP_END*60):
    """
    l, i, i_target are given in seconds
    assume: i_target >= keep_start + keep_end (returns -1 if not)
    assume: 0 < l <= i
    assume: i_target <= i
    """

    f = i - l
    keep_total = keep_start + keep_end
    ratio = (i_target - keep_total) / jnp.maximum(1, i - keep_total)

    case_0 = 1.0 * l
    case_1 = -1.0
    case_2 = 1.0 * i_target - f
    case_3 = 1.0 * l
    case_4 = keep_start + (l - keep_start) * ratio

    which = jnp.argmax(jnp.array([
        i == i_target,
        i_target < keep_total,
        f < keep_end,
        l < keep_start,
        True,
    ]))
    
    return jax.lax.select_n(which, case_0, case_1, case_2, case_3, case_4)


def _impute_li(l, i, available_intervals, available_ls):
    i_imputed = available_intervals[jnp.searchsorted(available_intervals, i)]
    l_imputed_frac = _impute_l(l, i, i_imputed)
    l_imputed = available_ls[jnp.searchsorted(available_ls, l_imputed_frac)]
    l_imputed = jnp.where(l_imputed_frac > 0, l_imputed, l_imputed_frac.astype(int))
    return l_imputed, i_imputed
