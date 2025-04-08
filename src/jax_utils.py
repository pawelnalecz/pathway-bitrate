import numpy as np

import jax
import jax.numpy as jnp


def jax_assert(x, msg):
    def error():
        raise ValueError(msg)

    jax.lax.cond(
        x,
        lambda: None,
        lambda: jax.debug.callback(error)
    )


# TODO(frdrc): these should be removed, since they assume slot=60s

BPH = 60 / np.log(2)

def nat_per_slot_to_bit_per_hour(x):
    return float(x * 60 / jnp.log(2))
