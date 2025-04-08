import matplotlib
from subplots_from_axsize import subplots_from_axsize

import jax.numpy as jnp

import sys 
from pathlib import Path
# sys.path.append(str(Path(__file__).parent.parent.resolve()))
sys.path.append(str(Path('.').resolve()))

from src import fig_style
from src.jax_dataloader import _impute_li

from config.local_config import OUTPUT_PATH

matplotlib.use("agg")
workdir: OUTPUT_PATH


# CONFIGS

# INCLUDE CORE RULES

# RULES


rule fig_S7:
    output:
        svg='figures/panels/figS7.svg',
        png='figures/panels/figS7.png',
    run:

        available_intervals = 60 * jnp.arange(5, 35+1)
        available_ls = 60 * jnp.arange(1, 35+1)
        i = 60 * 60
        ls = 60 * jnp.arange(1, 60+1)

        l_embeds = jnp.array([_impute_li(l, i, available_intervals, available_ls)[0] for l in ls])

        xlim = (-2, 60 + 2)
        ylim = (-2, 35 + 2)

        scale = 15
        fig, ax = subplots_from_axsize(
            axsize=(
                (xlim[1] - xlim[0]) / scale,
                (ylim[1] - ylim[0]) / scale
            )
        )

        ax.plot(ls / 60, l_embeds / 60, '-ok', markersize=5)
        ax.annotate('$interval_k = 60$ min', (0.5, 0.9), xycoords='axes fraction', va='center', ha='center')
        ax.set_xlabel('$last_k$ [min]')
        ax.set_ylabel('$last_k^*$ [min]')
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        fig.savefig(str(output.svg))
        fig.savefig(str(output.png), dpi=300)


