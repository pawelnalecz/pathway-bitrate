import matplotlib
from matplotlib.ticker import MultipleLocator
from subplots_from_axsize import subplots_from_axsize
import pandas as pd
import numpy as np

from generic._defaults import (
    DATA_MANAGER,
    WELLS_SELECTED,
    SET_TYPES,
)

from src import fig_style
from src.parameters import RESPONSIVENESS_DELAY
from config.local_config import OUTPUT_PATH

matplotlib.use("agg")
workdir: OUTPUT_PATH


# CONFIGS

well_id = '2024-11-27-BEAS2B--01s-0uM-rep1'

# INCLUDE CORE RULES

include: "generic/_core_rules.smk"
include: "generic/_measures.smk"

# RULES

rule fig_S2:
    input:
        tracks=f'cache/preprocessed/per_well/{well_id}/tracks_preprocessed.pkl.gz',
        response_reference=f'cache/response_amplitude/single/per_well/{well_id}/response-reference_q0_{RESPONSIVENESS_DELAY}.csv',
    output:
        svg='figures/panels/figS2.svg',
        png='figures/panels/figS2.png',
    resources:
        mem_gib=lambda wc, input: 2 * len(input),
    run:
        wells = ['2024-11-27-BEAS2B--01s-0uM-rep1']

        experiment = DATA_MANAGER.get_experiment(well_id)
        pulses = DATA_MANAGER.get_pulses(experiment)['time_in_seconds']

        fig, axs = subplots_from_axsize(axsize=(3, [1.6, 2.2, 1.6]), left=.8, top=0.4, right=.02, bottom=.8)


        track_id = 248 #94 # 3
        pulse_id = 40 #23#53
        xmin = pulses[pulse_id] - 2 * 60
        xmax = pulses[pulse_id] + 27 * 60

        pulse_duration = pulses[pulse_id + 1] - pulses[pulse_id]


        trks_preprocessed = pd.read_pickle(str(input['tracks']))
        trks_preprocessed['log_translocation'] = np.log(trks_preprocessed['translocation'])

        valid_timepoints = (
            trks_preprocessed['translocation'].gt(0) 
          & trks_preprocessed['I'].notna()
          & trks_preprocessed['log_translocation'].groupby('track_id').shift(-RESPONSIVENESS_DELAY).notna()
        )

        reference = trks_preprocessed[
            valid_timepoints
          & (trks_preprocessed['I'] > pulse_duration + RESPONSIVENESS_DELAY * 60)
            ].groupby('L')['log_translocation'].mean().loc[:pulse_duration + (RESPONSIVENESS_DELAY + 1) * 60 ]
            
        reference[0] = trks_preprocessed[trks_preprocessed['y'] == 1]['log_translocation'].mean()
        reference.sort_index(inplace=True)
        reference.name = 'reference'

    
        trks_preprocessed_slice = trks_preprocessed[
            (trks_preprocessed.index.get_level_values('time_in_seconds') >= xmin)
          & (trks_preprocessed.index.get_level_values('time_in_seconds') <= xmax)
          & valid_timepoints
        ]


        means = trks_preprocessed_slice.loc[track_id][['L', 'log_translocation']]#trks_preprocessed.groupby('time_in_seconds')[['L', 'log_translocation']].mean()
        # means = trks_preprocessed_slice.groupby('time_in_seconds')[['L', 'log_translocation']].mean()
        means['time_after_first_pulse'] = means.index - pulses[pulse_id]
        means = means.join(reference, on='time_after_first_pulse')


        # MEAN RESPONSE & REFERENCE

        # ax = axs[3]

        # means['log_translocation'].plot(
        #     ax=ax, lw=2, 
        #     color='k',
        #     label='mean log translocation',
        # )

        # alignment_term = means.loc[pulses[pulse_id + 1], 'log_translocation'] - means.loc[pulses[pulse_id + 1], 'reference']
        # (means['reference'] + alignment_term).plot(
        #     ax=ax, lw=2, 
        #     ls='--',
        #     color='k',
        #     alpha=.5,
        #     label='reference',
        # )

        # measurement_reference_time = pulses[pulse_id + 1]
        # measurement_time = pulses[pulse_id + 1] + RESPONSIVENESS_DELAY * 60
        # ax.annotate(
        #     '', 
        #     xy=(measurement_time, means.loc[measurement_time, 'log_translocation']),
        #     xytext=(measurement_time, means.loc[measurement_time, 'reference'] + alignment_term),
        #     arrowprops=dict(
        #         arrowstyle='<->',
        #         color='darkblue',
        #         lw=1.5,
        #     ),
        #     color='darkblue',
        # )
        
        # amplitude = means.loc[measurement_time, 'log_translocation'] - (means.loc[measurement_time, 'reference'] + alignment_term) 
        # ax.annotate(
        #     f'response\namplitude\n{amplitude:.3f}', 
        #     xy=(measurement_time + 60, (means.loc[measurement_time, 'log_translocation'] + means.loc[measurement_time, 'reference'] + alignment_term) / 2),
        #     fontweight='bold',
        #     color='darkblue',
        #     verticalalignment='center',
        # )


        # ylim = ax.get_ylim()
        # y_arrows = .7 * ylim[0] + .3 * ylim[1]

        # ax.annotate(
        #     '', 
        #     xy=(pulses[pulse_id], y_arrows),
        #     xytext=(pulses[pulse_id + 1], y_arrows),
        #     arrowprops=dict(
        #         arrowstyle='<->',
        #         color='darkblue',
        #         lw=1.5,
        #     ),
        #     color='darkblue',
        # )

        # ax.annotate(
        #     'interval I', 
        #     xy=((pulses[pulse_id] + pulses[pulse_id + 1]) / 2, y_arrows),
        #     color='darkblue',
        #     verticalalignment='bottom',
        #     horizontalalignment='center',
        # )


        # ax.annotate(
        #     '', 
        #     xy=(pulses[pulse_id + 1], y_arrows),
        #     xytext=(pulses[pulse_id + 1] + RESPONSIVENESS_DELAY * 60, y_arrows),
        #     arrowprops=dict(
        #         arrowstyle='<->',
        #         color='darkblue',
        #         lw=1.5,
        #     ),
        #     color='darkblue',
        # )

        # ax.annotate(
        #     f'{RESPONSIVENESS_DELAY} min', 
        #     xy=(pulses[pulse_id + 1] + (RESPONSIVENESS_DELAY * 60) / 2, y_arrows),
        #     color='darkblue',
        #     verticalalignment='bottom',
        #     horizontalalignment='center',
        # )
        
        
        # for pulse in pulses[pulses.between(xmin, xmax)]:
        #     ax.axvline(pulse, ls='--', alpha=.7, color='deepskyblue')

        
        # ax.set_ylabel('log ERKKTR translocation')

        # ax.legend()
        # ax.set_xlim(xmin, xmax)
        # ax.xaxis.set_major_locator(MultipleLocator(600))
        # ax.xaxis.set_minor_locator(MultipleLocator(300))
        # ax.xaxis.set_major_formatter(lambda x, pos: f"{x // 60:.0f}" if not x % 600 else '')
        # ax.set_xlabel('time [min]')


        # MEAN RESPONSE

        ax = axs[0]
        

        means['log_translocation'].plot(
            ax=ax, lw=1, 
            color='k',
            alpha=.7,
            label='single cell trajectory',
        )


        ylim = ax.get_ylim()
        y_arrows = .82 * ylim[0] + .18 * ylim[1]

        measurement_reference_time = pulses[pulse_id + 1]
        measurement_time = pulses[pulse_id + 1] + RESPONSIVENESS_DELAY * 60
        ax.annotate(
            '', 
            xy=(measurement_time, means.loc[measurement_time, 'log_translocation']),
            xytext=(measurement_time, means.loc[measurement_reference_time, 'log_translocation']),
            arrowprops=dict(
                arrowstyle='<->',
                color='darkblue',
                lw=1.5,
                shrinkA=0,
            ),
            color='darkblue',
        )

        ax.annotate(
            '', 
            xy=(measurement_reference_time, means.loc[measurement_reference_time, 'log_translocation']),
            xytext=(measurement_time, means.loc[measurement_reference_time, 'log_translocation']),
            arrowprops=dict(
                arrowstyle='-',
                color='darkblue',
                ls=':'
            ),
            color='darkblue',
            alpha=.3,
        )
        
        amplitude = means.loc[measurement_time, 'log_translocation'] - means.loc[measurement_reference_time, 'log_translocation'] 
        ax.annotate(
            f'raw\nresponse\namplitude', 
            xy=(measurement_time + 30, (means.loc[measurement_time, 'log_translocation'] + means.loc[measurement_reference_time, 'log_translocation']) / 2 - (ylim[1] - ylim[0]) * .2),
            fontweight='bold',
            color='darkblue',
            verticalalignment='center',
        )



        ax.annotate(
            '', 
            xy=(pulses[pulse_id], y_arrows),
            xytext=(pulses[pulse_id + 1], y_arrows),
            arrowprops=dict(
                arrowstyle='<->',
                color='darkblue',
                lw=1.5,
            ),
            color='darkblue',
        )

        ax.annotate(
            'interval L', 
            xy=((pulses[pulse_id] + pulses[pulse_id + 1]) / 2, y_arrows),
            color='darkblue',
            verticalalignment='bottom',
            horizontalalignment='center',
        )


        ax.annotate(
            '', 
            xy=(pulses[pulse_id + 1], y_arrows),
            xytext=(pulses[pulse_id + 1] + RESPONSIVENESS_DELAY * 60, y_arrows),
            arrowprops=dict(
                arrowstyle='<->',
                color='darkblue',
                lw=1.5,
            ),
            color='darkblue',
        )

        ax.annotate(
            f'{RESPONSIVENESS_DELAY} min', 
            xy=(pulses[pulse_id + 1] + (RESPONSIVENESS_DELAY * 60) / 2, y_arrows),
            color='darkblue',
            verticalalignment='bottom',
            horizontalalignment='center',
        )
        



        for pulse in pulses[pulses.between(xmin, xmax)]:
            ax.axvline(pulse, ls='--', alpha=.7, color='deepskyblue')
        ax.annotate('pulse n−1 ', (pulses[pulse_id    ] - 30, ylim[1]), rotation='vertical', horizontalalignment='right', verticalalignment='top', color='deepskyblue', fontweight='bold')
        ax.annotate('pulse n   ', (pulses[pulse_id + 1] - 30, ylim[1]), rotation='vertical', horizontalalignment='right', verticalalignment='top', color='deepskyblue', fontweight='bold')

        ax.legend()
        ax.set_ylabel('log ERK-KTR translocation')
        ax.legend(loc='lower right')

        ax.set_xlim(xmin, xmax)
        ax.xaxis.set_major_locator(MultipleLocator(600))
        ax.xaxis.set_minor_locator(MultipleLocator(300))
        ax.xaxis.set_major_formatter(lambda x, pos: f"{x // 60:.0f}" if not x % 600 else '')
        ax.set_xlabel('time [min]')

        # REFERENCE

        ax = axs[2]
        
        reference.plot(
            ax=ax, lw=2, 
            ls='--',
            color='k',
            alpha=.5,
            label='reference',
        )

        measurement_reference_time = pulse_duration
        measurement_time = pulse_duration + RESPONSIVENESS_DELAY * 60
        ax.annotate(
            '', 
            xy=(measurement_time, reference.loc[measurement_time]),
            xytext=(measurement_time, reference.loc[measurement_reference_time]),
            arrowprops=dict(
                arrowstyle='<->',
                color='darkblue',
                lw=1.5,
                shrinkA=0,
            ),
            color='darkblue',
        )

        ax.annotate(
            '', 
            xy=(measurement_reference_time, reference.loc[measurement_reference_time]),
            xytext=(measurement_time, reference.loc[measurement_reference_time]),
            arrowprops=dict(
                arrowstyle='-',
                color='darkblue',
                ls=':',
                shrinkA=0,
            ),
            color='darkblue',
            alpha=.3,
        )
        
        reference_amplitude = reference.loc[measurement_time] - reference.loc[measurement_reference_time] 
        ax.annotate(
            f'reference\ncurve\nchange', 
            xy=(measurement_time + 60, (reference.loc[measurement_time] + reference.loc[measurement_reference_time]) / 2),
            fontweight='bold',
            color='darkblue',
            verticalalignment='center',
        )


        ylim = ax.get_ylim()
        y_arrows = .95 * ylim[0] + .05 * ylim[1]

        ax.annotate(
            f'reponse amplitude \n= raw response amplitude\n− reference curve change ', 
            xy=(5 * 60, y_arrows),
            color='darkblue',
            verticalalignment='bottom',
            horizontalalignment='left',
        )
        

        ax.axvline(0, ls='--', alpha=.7, color='deepskyblue')
        ax.axvline(pulses[pulse_id + 1] - pulses[pulse_id], color='k', ls=':', alpha=.3)
        ax.annotate('pulse ', (-30, ylim[1]), rotation='vertical', horizontalalignment='right', verticalalignment='top', color='deepskyblue', fontweight='bold')
 
        ax.set_ylabel('log ERK-KTR translocation')
        ax.legend()

        ax.set_xlim(xmin - pulses[pulse_id], xmax - pulses[pulse_id])
        ax.xaxis.set_major_locator(MultipleLocator(600))
        ax.xaxis.set_minor_locator(MultipleLocator(300))
        ax.xaxis.set_major_formatter(lambda x, pos: f"{x // 60:.0f}" if not x % 600 else '')
        ax.set_xlabel('time after pulse [min]')


        # REFERENCE WITH TRAJECTORIES

        ax = axs[1]
        

        trks_preprocessed['last_pulse'] = trks_preprocessed.index.get_level_values('time_in_seconds') - trks_preprocessed['L'] * (1 - trks_preprocessed['y'])
        trks_preprocessed['L_from_zero'] = trks_preprocessed['L'] * (1 - trks_preprocessed['y'])

        trks_used_for_reference = trks_preprocessed[
            valid_timepoints
          & (
                # (trks_preprocessed['I'] > pulse_duration + RESPONSIVENESS_DELAY * 60) 
               (trks_preprocessed.groupby('track_id')['I'].shift(-1) > pulse_duration + RESPONSIVENESS_DELAY * 60)
            )
        ]

        trks_used_for_reference.loc[range(10)].groupby(['track_id', 'last_pulse']).plot(
            'L_from_zero', 'log_translocation', 
            ax=ax, lw=1, 
            ls='-',
            color='k',
            alpha=.1,
            label='_',
        )

        reference.plot(
            ax=ax, lw=2, 
            ls='--',
            color='k',
            alpha=.5,
            label='reference',
        )



        ylim = ax.get_ylim()
        y_arrows = .1 * ylim[0] + .9 * ylim[1]

        ax.axvline(0, ls='--', alpha=.7, color='deepskyblue')
        ax.annotate('pulse ', (-30, ylim[1]), rotation='vertical', horizontalalignment='right', verticalalignment='top', color='deepskyblue', fontweight='bold')
        # ax.axvline(pulses[pulse_id + 1] - pulses[pulse_id], color='k', ls=':', alpha=.3)

        ax.annotate(
            '', 
            xy=(0, y_arrows),
            xytext=(pulse_duration + RESPONSIVENESS_DELAY * 60, y_arrows),
            arrowprops=dict(
                arrowstyle='-',
                color='grey',
                lw=1.5,
            ),
            color='grey',
        )

        ax.annotate(
            'no pulse in L + 7 min', 
            xy=((pulse_duration + RESPONSIVENESS_DELAY * 60) / 2, y_arrows),
            color='darkblue',
            verticalalignment='bottom',
            horizontalalignment='center',
        )


        ax.legend(loc='lower right')
        # ax.get_legend().set_visible(False)

        ax.set_ylabel('log ERK-KTR translocation')

        ax.set_xlim(xmin - pulses[pulse_id], xmax - pulses[pulse_id])
        ax.xaxis.set_major_locator(MultipleLocator(600))
        ax.xaxis.set_minor_locator(MultipleLocator(300))
        ax.xaxis.set_major_formatter(lambda x, pos: f"{x // 60:.0f}" if not x % 600 else '')
        ax.set_xlabel('time after pulse [min]')



        ax = axs[0]

        # reference2 = pd.read_csv(input['response_reference'], index_col='L')['response_reference']
        
        # means2 = trks_preprocessed_slice.loc[track_id][['L', 'log_translocation_ratio']]#trks_preprocessed.groupby('time_in_seconds')[['L', 'log_translocation_ratio']].mean()
        # means2 = trks_preprocessed.groupby('time_in_seconds')[['L', 'log_translocation_ratio']].mean()
        # means2['time_after_first_pulse'] = means2.index - pulses[pulse_id]
        # means2 = means2.join(reference2, on='time_after_first_pulse')

        # means2['log_translocation_ratio'].plot(
        #     ax=ax, lw=2, 
        #     color='olive',
        # )

        # alignment_term = 0#means.loc[pulses[pulse_id + 1], 'log_translocation'] - means.loc[pulses[pulse_id + 1], 'reference']
        
        # (means2['response_reference'] + alignment_term).plot(
        #     ax=ax, lw=2, 
        #     ls='--',
        #     color='olive',
        #     alpha=.5,
        # )


        # for pulse in pulses[pulses.between(xmin, xmax)]:
        #     ax.axvline(pulse, ls='--', alpha=.7, color='deepskyblue')

        # measurement_time = pulses[pulse_id + 1]
        # ax.annotate(
        #     '', 
        #     xy=(measurement_time, means2.loc[measurement_time, 'log_translocation_ratio']),
        #     xytext=(measurement_time, means2.loc[measurement_time, 'response_reference'] + alignment_term),
        #     arrowprops=dict(
        #         arrowstyle='<->',
        #         color='darkblue',
        #         lw=1.5,
        #         shrinkA=0,
        #     ),
        #     color='darkblue',
        # )

        # amplitude2 = means2.loc[measurement_time, 'log_translocation_ratio'] - (means2.loc[measurement_time, 'response_reference'] + alignment_term)
        # ax.annotate(
        #     f'response\namplitude\n{amplitude2:.3f}', 
        #     xy=(measurement_time + 60, (means2.loc[measurement_time, 'log_translocation_ratio'] + means2.loc[measurement_time, 'response_reference'] + alignment_term) / 2),
        #     fontweight='bold',
        #     color='darkblue',
        #     verticalalignment='center',
        # )



        # ax.set_ylabel(f'ERKKTR translocation\nlog fold change in next {RESPONSIVENESS_DELAY} min')

        # ax.set_xlim(xmin, xmax)
        # ax.xaxis.set_major_locator(MultipleLocator(600))
        # ax.xaxis.set_minor_locator(MultipleLocator(300))
        # ax.xaxis.set_major_formatter(lambda x, pos: f"{x // 60:.0f}" if not x % 600 else '')
        # ax.set_xlabel('time [min]')
            

        fig.savefig(str(output.svg))
        fig.savefig(str(output.png), dpi=300)

