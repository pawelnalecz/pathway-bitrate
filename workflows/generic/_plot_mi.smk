from subplots_from_axsize import subplots_from_axsize
import pandas as pd

from src.fig_layout import val_name_pos_list, row_to_pos


rule plot_mi:
    input:
        mi_all='{plot_dir}/mi_all.csv'
    output:
        '{plot_dir}/mi_{dataset_id}_{model_id}_{train_id}_{test_id}.png'
    run:
        mi_all = pd.read_csv(input.mi_all)

        train_ids = wildcards.train_id.split('+')
        max_train_ids = len([col for col in mi_all.columns if col.startswith('train_id_')])
       
        mi_plot = mi_all[
            (mi_all['dataset_id'] == wildcards.dataset_id)
          & (mi_all['model_id'] == wildcards.model_id)
          & (mi_all[[f'train_id_{it}' for it in range(len(train_ids))]] == train_ids).all(axis=1)
          & (mi_all[[f'train_id_{it}' for it in range(len(train_ids), max_train_ids)]].isna()).all(axis=1)
          & (mi_all['test_id'] == wildcards.test_id)
        ].copy()

        title = '; '.join([
            wildcards.dataset_id,
            wildcards.model_id,
            wildcards.train_id,
            wildcards.test_id,
        ])

        mi_plot['pos'] = mi_plot.apply(row_to_pos, axis=1)

        fig, ax = subplots_from_axsize(axsize=(6, 3), top=0.4)

        for experiment, mi_plot_exp in mi_plot.groupby('experiment'):
            ax.plot(
                mi_plot_exp['pos'],
                mi_plot_exp['mi_ce'],
                'o',
                alpha=0.5,
                label=experiment,
            )

        ax.legend()
        ax.set_xlim(0.5, 13.0)
        bottom = min(-0.5, mi_plot['mi_ce'].min())
        top = max(12.0, mi_plot['mi_ce'].max())
        ax.set_ylim(bottom, top)
        ax.grid(color='k', alpha=0.5, ls=':')
        ax.set_xticks(*list(zip(*[(pos, name) for _, name, pos in val_name_pos_list])))
        ax.set_ylabel('bitrate [bit / hour]')
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_title(title)

        fig.savefig(str(output))
