import seaborn as sns
from subplots_from_axsize import subplots_from_axsize
from scipy.special import expit
import pandas as pd


rule li_mean_p_network:
    input:
        predictions='cache/predictions/{per}/{well_or_set_id}/predictions_{dataset_id}_{model_id}_{train_id}_{test_id}.csv.gz'
    output:
        'cache/network_evaluation/single/{per}/{well_or_set_id}/li-mean-p-{network}_{dataset_id}_{model_id}_{train_id}_{test_id}.csv'
    run:
        predictions = pd.read_csv(input.predictions)

        if wildcards.network == 'network':
            predictions['p_network'] = expit(predictions['p_network_logit'])
            field = 'p_network'
        elif wildcards.network == 'network-logit':
            field = 'p_network_logit'
        else:
            raise ValueError(f"Wildcard 'network' must be either 'network' or 'network-logit', not {wildcards.network}")
        
        predictions['L'] = (predictions['L'] / 60).astype(int)
        predictions['I'] = (predictions['I'] / 60).astype(int)

        li_mean_p_network = (
            predictions.groupby(['L', 'I'])[field]
            .mean()
            .reset_index(level=-1).pivot(columns='I', values=field)
        )

        li_mean_p_network = li_mean_p_network[li_mean_p_network.index.isin(li_mean_p_network.columns)]

        li_mean_p_network.to_csv(str(output))


rule nn_predictions_heatmap:
    input:
        # predictions='cache/predictions/{per}/{well_or_set_id}/predictions_{dataset_id}_{model_id}_{train_id}_{test_id}.csv.gz'
        'cache/network_evaluation/single/{per}/{well_or_set_id}/li-mean-p-network_{dataset_id}_{model_id}_{train_id}_{test_id}.csv'
    output:
        '{plot_dir}/single/{per}/{well_or_set_id}/nn-predictions-heatmap_{dataset_id}_{model_id}_{train_id}_{test_id}.png'
    run:
        fig, (ax, cbar_ax) = subplots_from_axsize(axsize=([3, .1], 3), top=.8)

        li_mean_p_network = pd.read_csv(str(input)).set_index('L')

        sns.heatmap(li_mean_p_network, center=0.5, square=True, vmin=0, vmax=1, ax=ax, cbar_ax=cbar_ax)

        fig.suptitle(f'{wildcards.well_or_set_id} -{wildcards.test_id}\n{wildcards.dataset_id}; {wildcards.model_id};\n {wildcards.train_id}')
        dpi = 300
        fig.savefig(str(output), bbox_inches='tight', dpi=dpi)


rule nn_predictions_diagonal:
    input:
        # predictions='cache/predictions/{per}/{well_or_set_id}/predictions_{dataset_id}_{model_id}_{train_id}_{test_id}.csv.gz'
        'cache/network_evaluation/single/{per}/{well_or_set_id}/li-mean-p-network_{dataset_id}_{model_id}_{train_id}_{test_id}.csv'
    output:
        '{plot_dir}/single/{per}/{well_or_set_id}/nn-predictions-diagonal_{dataset_id}_{model_id}_{train_id}_{test_id}.png'
    run:
        fig, ax = subplots_from_axsize(axsize=(6, 3), top=.8)

        li_mean_p_network = pd.read_csv(str(input), index_col='L')
        li_mean_p_network.columns = map(int, li_mean_p_network.columns)
        intervals = li_mean_p_network.columns
        assert all(i in li_mean_p_network.index for i in intervals), "Not all intervals found as L"

        li_mean_p_network_diagonal = pd.Series([li_mean_p_network.loc[i, i] for i in intervals], index=intervals)

        # sns.heatmap(li_mean_mlp_logit, center=0, square=True, vmin=-3.5, vmax=3.5, ax=ax, cbar_ax=cbar_ax)
        ax.plot(intervals, li_mean_p_network_diagonal, color='k', marker='o', ms=3)

        ax.set_ylim(0, 1)

        fig.suptitle(f'{wildcards.well_or_set_id} -{wildcards.test_id}\n{wildcards.dataset_id}; {wildcards.model_id};\n {wildcards.train_id}')
        dpi = 300
        fig.savefig(str(output), bbox_inches='tight', dpi=dpi)
