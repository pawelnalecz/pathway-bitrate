import os
import jax

os.environ["JAX_PLATFORM_NAME"] = "cpu"
jax.config.update("jax_enable_x64", True)
