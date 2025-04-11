import matplotlib
from matplotlib import ticker
from subplots_from_axsize import subplots_from_axsize

from generic._defaults import (
    DATA_MANAGER,
    WELLS_SELECTED,
    SET_TYPES,
    DATASET_IDS,
    MODEL_IDS,
    TRAIN_IDS,
    TEST_IDS,
)

from src import jax_plots
from src.jax_evaluation import cross_entropy_from_logit

from config.local_config import OUTPUT_PATH

matplotlib.use("agg")
workdir: OUTPUT_PATH

# INCLUDE CORE RULES

include: "generic/_combine_plots.smk"
include: "generic/_core_rules.smk"


# RULES

rule all:
    input:
        expand(
            'plot_predictions/combined/per_experiment/{experiment}/{plot_type}_{dataset_id}_{model_id}_{train_id}_{test_id}.png',
            plot_type=[
                'predictions-by-ly-mean',
                'predictions-by-ly-hist',
                'mi-by-time-mean',
            ],
            experiment=WELLS_SELECTED['experiment'].unique(),
            dataset_id=DATASET_IDS,
            model_id=MODEL_IDS,
            train_id=TRAIN_IDS,
            test_id=TEST_IDS,
        )


rule plot_predictions_by_ly_mean:
    input:
        predictions='cache/predictions/{per}/{well_or_set_id}/predictions_{dataset_id}_{model_id}_{train_id}_{test_id}.csv.gz',
        protocol=inputs_test_protocol,
    output:
        'plot_predictions/single/{per}/{well_or_set_id}/predictions-by-ly-mean_{dataset_id}_{model_id}_{train_id}_{test_id}.png'
    resources:
        mem_gib=1
    run:
        protocol = jax_protocol.Protocol(path=input.protocol)
        predictions = pd.read_csv(input.predictions)

        fig, ax = subplots_from_axsize(axsize=(4, 3), top=0.6, right=0.4)
        jax_plots.plot_predictions_by_ly_mean(ax, protocol, predictions)
        ax.set_title(f'{wildcards.well_or_set_id} -{wildcards.test_id}\n{wildcards.dataset_id}; {wildcards.model_id}; {wildcards.train_id};')
        fig.savefig(str(output), dpi=300)


rule plot_predictions_by_ly_hist:
    input:
        predictions='cache/predictions/{per}/{well_or_set_id}/predictions_{dataset_id}_{model_id}_{train_id}_{test_id}.csv.gz',
        protocol=inputs_test_protocol,
    output:
        'plot_predictions/single/{per}/{well_or_set_id}/predictions-by-ly-hist_{dataset_id}_{model_id}_{train_id}_{test_id}.png'
    resources:
        mem_gib=1
    run:
        protocol = jax_protocol.Protocol(path=input.protocol)
        predictions = pd.read_csv(input.predictions)

        fig, ax = subplots_from_axsize(axsize=(8, 3), top=0.6, right=0.4)
        jax_plots.plot_predictions_by_ly_hist(ax, protocol, predictions)
        ax.set_title(f'{wildcards.well_or_set_id} -{wildcards.test_id}\n{wildcards.dataset_id}; {wildcards.model_id}; {wildcards.train_id};')
        fig.savefig(str(output), dpi=300)


rule plot_mi_by_time_mean:
    input:
        predictions='cache/predictions/per_well/{well_id}/predictions_{dataset_id}_{model_id}_{train_id}_{test_id}.csv.gz',
        protocol=inputs_test_protocol,
    output:
        'plot_predictions/single/per_well/{well_id}/mi-by-time-mean_{dataset_id}_{model_id}_{train_id}_{test_id}.png'
    resources:
        mem_gib=1
    run:
        well_info = DATA_MANAGER.get_well_info(wildcards.well_id)
        experiment = well_info['experiment']

        protocol = jax_protocol.Protocol(path=input.protocol)
        predictions = pd.read_csv(input.predictions)
        pulses = DATA_MANAGER.get_pulses(experiment)
        seconds_per_timepoint = DATA_MANAGER.get_seconds_per_timepoint(experiment)

        predictions['time_in_hours'] = predictions['time_in_seconds'] / 60 / 60
        time_end = predictions['time_in_hours'].max() + 5 / 60

        predictions['pe'] = cross_entropy_from_logit(predictions['p_prior_logit'], predictions['y']) / np.log(2)
        predictions['re_cross'] = cross_entropy_from_logit(predictions['p_predicted_logit'], predictions['y']) / np.log(2)
        predictions['mi_cross'] = predictions['pe'] - predictions['re_cross']

        fig, ax = subplots_from_axsize(
            axsize=(time_end / 3, 2),
            left=0.75,
	        right=0.75,
            top=0.5,
        )

        slots_per_hour = 60 * 60 / seconds_per_timepoint
        mi_cross_rolling_3h_mean = (
            predictions.groupby('time_in_hours')['mi_cross'].sum().rolling(180, center=True).sum()
          / predictions.groupby('time_in_hours').size().rolling(180, center=True).sum()
        )
        ax.plot(
            slots_per_hour * mi_cross_rolling_3h_mean,
            color='darkgreen',
            label='MI_cross (3h average)',
        )
        ax.set_ylim(bottom=0.0)
        ax.set_ylabel('MI smoothed [bit / hour]')

        ax_count = ax.twinx()
        ax_count.plot(
            predictions.groupby('time_in_hours').size(),
            color='grey',
            label='weight',
        )
        ax_count.set_ylabel('number of tracks')
        ax_count.set_ylim(bottom=0)

        #ax.plot(
        #    predictions.groupby('time_in_hours')['mi_cross'].mean(),
        #    color='darkgreen',
        #    alpha=0.5,
        #)
        #ax.axhline(0.0, color='gray', alpha=0.2)
        #ax.set_ylabel('MI [bit / slot]')

        # plot pulses
        for _, pulse in pulses.iterrows():
            time_in_hours = pulse.time_in_seconds / 60 / 60

            if time_in_hours > time_end:
                continue

            ax.axvline(
                time_in_hours,
                color='b' if pulse.valid else 'r',
                ls='--',
                lw=0.5,
                alpha=0.5,
            )

        ax.set_xlim(0.0, time_end)
        ax.set_xlabel('time [hour]')

        ax.spines[['top', 'right']].set_visible(False)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1 / 6))

        ax.set_title(f"{wildcards.well_id} -{wildcards.test_id}\n{wildcards.dataset_id}; {wildcards.model_id}; {wildcards.train_id}")

        fig.savefig(str(output), dpi=300)


